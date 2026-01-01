#!/usr/bin/env python3
"""
LocalMod Discord Bot - Automatic content moderation for Discord servers.

Features:
- Real-time text moderation (toxicity, spam, prompt injection, NSFW)
- Image moderation (NSFW detection)
- PII detection and auto-redaction
- Configurable actions (warn, delete, timeout, ban)
- Logging to a mod channel

Setup:
1. Create a Discord bot at https://discord.com/developers/applications
2. Enable MESSAGE CONTENT INTENT in Bot settings
3. Set DISCORD_BOT_TOKEN environment variable
4. Invite bot with permissions: Manage Messages, Kick Members, Ban Members, Moderate Members

Usage:
    export DISCORD_BOT_TOKEN=your_token_here
    python examples/discord_bot.py
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import discord
    from discord import app_commands
    from discord.ext import commands
except ImportError:
    print("❌ discord.py not installed. Run: pip install discord.py")
    sys.exit(1)

from localmod import SafetyPipeline
from localmod.classifiers.pii import PIIDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('localmod_bot')


# =============================================================================
# Configuration
# =============================================================================

class BotConfig:
    """Bot configuration - customize these for your server."""
    
    # Moderation settings
    ENABLED_CLASSIFIERS = ["toxicity", "spam", "prompt_injection", "nsfw", "pii"]
    
    # Thresholds for different actions (confidence levels)
    WARN_THRESHOLD = 0.5      # Just log, no action
    DELETE_THRESHOLD = 0.7    # Delete the message
    TIMEOUT_THRESHOLD = 0.85  # Delete + timeout user (5 min)
    BAN_THRESHOLD = 0.95      # Extreme cases only
    
    # Timeout duration (in minutes)
    TIMEOUT_DURATION = 5
    
    # Channels to ignore (add channel IDs here)
    IGNORED_CHANNELS: List[int] = []
    
    # Roles that bypass moderation (add role IDs here)
    BYPASS_ROLES: List[int] = []
    
    # Log channel name (will be found or created)
    LOG_CHANNEL_NAME = "localmod-logs"
    
    # Enable image moderation
    MODERATE_IMAGES = True
    
    # Auto-redact PII in messages
    AUTO_REDACT_PII = False  # If True, edits message to redact PII


# =============================================================================
# Bot Setup
# =============================================================================

class LocalModBot(commands.Bot):
    """Discord bot with LocalMod integration."""
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True  # Required for reading messages
        intents.members = True  # Required for member management
        
        super().__init__(
            command_prefix="!mod ",
            intents=intents,
            description="LocalMod - AI Content Moderation Bot"
        )
        
        # Initialize LocalMod pipeline
        self.pipeline: Optional[SafetyPipeline] = None
        self.pii_detector: Optional[PIIDetector] = None
        self.image_classifier = None
        
        # Stats tracking
        self.stats = {
            "messages_scanned": 0,
            "messages_flagged": 0,
            "messages_deleted": 0,
            "users_timed_out": 0,
            "images_scanned": 0,
            "images_flagged": 0,
        }
    
    async def setup_hook(self):
        """Called when bot is starting up."""
        logger.info("🔧 Loading LocalMod classifiers...")
        
        # Load in a thread to not block
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_classifiers)
        
        logger.info("✅ LocalMod ready!")
        
        # Sync slash commands
        await self.tree.sync()
    
    def _load_classifiers(self):
        """Load LocalMod classifiers (runs in thread)."""
        self.pipeline = SafetyPipeline(
            classifiers=BotConfig.ENABLED_CLASSIFIERS,
            device="cpu"  # Use CPU for stability
        )
        
        # Pre-load PII detector
        self.pii_detector = PIIDetector()
        self.pii_detector.load()
        
        # Load image classifier if enabled
        if BotConfig.MODERATE_IMAGES:
            try:
                from localmod.classifiers.nsfw_image import ImageNSFWClassifier
                self.image_classifier = ImageNSFWClassifier(device="cpu")
                self.image_classifier.load()
                logger.info("🖼️ Image moderation enabled")
            except Exception as e:
                logger.warning(f"⚠️ Image moderation disabled: {e}")
                self.image_classifier = None
    
    async def on_ready(self):
        """Called when bot is connected and ready."""
        logger.info(f"🤖 Logged in as {self.user} (ID: {self.user.id})")
        logger.info(f"📊 Serving {len(self.guilds)} server(s)")
        
        # Set status
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="for bad content | /modstats"
            )
        )


# =============================================================================
# Message Moderation
# =============================================================================

bot = LocalModBot()


@bot.event
async def on_message(message: discord.Message):
    """Handle incoming messages."""
    # Ignore bot messages
    if message.author.bot:
        return
    
    # Ignore DMs
    if not message.guild:
        return
    
    # Check if channel is ignored
    if message.channel.id in BotConfig.IGNORED_CHANNELS:
        return
    
    # Check if user has bypass role
    if any(role.id in BotConfig.BYPASS_ROLES for role in message.author.roles):
        return
    
    # Process commands first
    await bot.process_commands(message)
    
    # Skip if no content
    if not message.content.strip() and not message.attachments:
        return
    
    # Moderate text content
    if message.content.strip():
        await moderate_text(message)
    
    # Moderate image attachments
    if BotConfig.MODERATE_IMAGES and message.attachments:
        await moderate_images(message)


async def moderate_text(message: discord.Message):
    """Moderate text content in a message."""
    if not bot.pipeline:
        return
    
    bot.stats["messages_scanned"] += 1
    
    # Run analysis
    try:
        report = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: bot.pipeline.analyze(message.content)
        )
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return
    
    if not report.flagged:
        return
    
    bot.stats["messages_flagged"] += 1
    
    # Get highest severity result
    max_result = max(report.results, key=lambda r: r.confidence if r.flagged else 0)
    confidence = max_result.confidence
    classifier = max_result.classifier
    
    # Determine action based on confidence
    action = determine_action(confidence)
    
    # Log to mod channel
    await log_moderation(message, report, action)
    
    # Take action
    if action == "delete" or action == "timeout":
        try:
            await message.delete()
            bot.stats["messages_deleted"] += 1
        except discord.Forbidden:
            logger.warning("Missing permission to delete message")
    
    if action == "timeout":
        try:
            await message.author.timeout(
                timedelta(minutes=BotConfig.TIMEOUT_DURATION),
                reason=f"LocalMod: {classifier} detected ({confidence:.0%})"
            )
            bot.stats["users_timed_out"] += 1
        except discord.Forbidden:
            logger.warning("Missing permission to timeout user")
    
    # Warn user via DM (optional)
    if action in ["delete", "timeout"]:
        try:
            await message.author.send(
                f"⚠️ Your message in **{message.guild.name}** was flagged for **{classifier}** "
                f"(confidence: {confidence:.0%}).\n\n"
                f"Please follow the server rules. Repeated violations may result in a ban."
            )
        except discord.Forbidden:
            pass  # User has DMs disabled


async def moderate_images(message: discord.Message):
    """Moderate image attachments."""
    if not bot.image_classifier:
        return
    
    for attachment in message.attachments:
        # Check if it's an image
        if not attachment.content_type or not attachment.content_type.startswith("image/"):
            continue
        
        bot.stats["images_scanned"] += 1
        
        try:
            # Download and analyze
            image_bytes = await attachment.read()
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: bot.image_classifier.predict(image_bytes)
            )
            
            if result.flagged:
                bot.stats["images_flagged"] += 1
                
                # Delete message with NSFW image
                try:
                    await message.delete()
                    bot.stats["messages_deleted"] += 1
                    
                    # Log
                    await log_image_moderation(message, result)
                    
                    # Warn user
                    await message.author.send(
                        f"⚠️ Your image in **{message.guild.name}** was flagged as NSFW "
                        f"(confidence: {result.confidence:.0%}).\n\n"
                        f"Please follow the server rules."
                    )
                except discord.Forbidden:
                    pass
                
        except Exception as e:
            logger.error(f"Image analysis error: {e}")


def determine_action(confidence: float) -> str:
    """Determine what action to take based on confidence."""
    if confidence >= BotConfig.BAN_THRESHOLD:
        return "ban"  # Not implemented for safety
    elif confidence >= BotConfig.TIMEOUT_THRESHOLD:
        return "timeout"
    elif confidence >= BotConfig.DELETE_THRESHOLD:
        return "delete"
    elif confidence >= BotConfig.WARN_THRESHOLD:
        return "warn"
    return "none"


async def log_moderation(message: discord.Message, report, action: str):
    """Log moderation action to mod channel."""
    log_channel = discord.utils.get(message.guild.channels, name=BotConfig.LOG_CHANNEL_NAME)
    
    if not log_channel:
        return
    
    # Build embed
    color = {
        "none": discord.Color.green(),
        "warn": discord.Color.yellow(),
        "delete": discord.Color.orange(),
        "timeout": discord.Color.red(),
        "ban": discord.Color.dark_red(),
    }.get(action, discord.Color.gray())
    
    embed = discord.Embed(
        title=f"🛡️ Content Flagged",
        color=color,
        timestamp=datetime.utcnow()
    )
    
    embed.add_field(name="User", value=f"{message.author.mention} ({message.author.id})", inline=True)
    embed.add_field(name="Channel", value=message.channel.mention, inline=True)
    embed.add_field(name="Action", value=action.upper(), inline=True)
    
    # Add flagged classifiers
    flagged = [r for r in report.results if r.flagged]
    for result in flagged[:3]:  # Limit to 3
        embed.add_field(
            name=f"📊 {result.classifier}",
            value=f"{result.confidence:.1%} ({result.severity.value})",
            inline=True
        )
    
    # Truncate content for display
    content = message.content[:500] + "..." if len(message.content) > 500 else message.content
    embed.add_field(name="Content", value=f"```{content}```", inline=False)
    
    embed.set_footer(text=f"LocalMod v{report.processing_time_ms:.0f}ms")
    
    try:
        await log_channel.send(embed=embed)
    except discord.Forbidden:
        logger.warning("Missing permission to send to log channel")


async def log_image_moderation(message: discord.Message, result):
    """Log image moderation to mod channel."""
    log_channel = discord.utils.get(message.guild.channels, name=BotConfig.LOG_CHANNEL_NAME)
    
    if not log_channel:
        return
    
    embed = discord.Embed(
        title="🖼️ NSFW Image Detected",
        color=discord.Color.red(),
        timestamp=datetime.utcnow()
    )
    
    embed.add_field(name="User", value=f"{message.author.mention}", inline=True)
    embed.add_field(name="Channel", value=message.channel.mention, inline=True)
    embed.add_field(name="Confidence", value=f"{result.confidence:.1%}", inline=True)
    
    try:
        await log_channel.send(embed=embed)
    except discord.Forbidden:
        pass


# =============================================================================
# Slash Commands
# =============================================================================

@bot.tree.command(name="modstats", description="View moderation statistics")
async def modstats(interaction: discord.Interaction):
    """Show moderation statistics."""
    embed = discord.Embed(
        title="📊 LocalMod Statistics",
        color=discord.Color.blue(),
        timestamp=datetime.utcnow()
    )
    
    embed.add_field(name="Messages Scanned", value=f"{bot.stats['messages_scanned']:,}", inline=True)
    embed.add_field(name="Messages Flagged", value=f"{bot.stats['messages_flagged']:,}", inline=True)
    embed.add_field(name="Messages Deleted", value=f"{bot.stats['messages_deleted']:,}", inline=True)
    embed.add_field(name="Users Timed Out", value=f"{bot.stats['users_timed_out']:,}", inline=True)
    embed.add_field(name="Images Scanned", value=f"{bot.stats['images_scanned']:,}", inline=True)
    embed.add_field(name="Images Flagged", value=f"{bot.stats['images_flagged']:,}", inline=True)
    
    # Calculate flag rate
    if bot.stats["messages_scanned"] > 0:
        rate = bot.stats["messages_flagged"] / bot.stats["messages_scanned"] * 100
        embed.add_field(name="Flag Rate", value=f"{rate:.2f}%", inline=True)
    
    embed.set_footer(text="LocalMod - Offline AI Moderation")
    
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="analyze", description="Analyze text for safety issues")
@app_commands.describe(text="The text to analyze")
async def analyze(interaction: discord.Interaction, text: str):
    """Manually analyze text."""
    if not bot.pipeline:
        await interaction.response.send_message("❌ LocalMod not loaded yet", ephemeral=True)
        return
    
    await interaction.response.defer(ephemeral=True)
    
    report = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: bot.pipeline.analyze(text)
    )
    
    embed = discord.Embed(
        title="🔍 Analysis Results",
        color=discord.Color.red() if report.flagged else discord.Color.green(),
        description=f"**Status:** {'⚠️ FLAGGED' if report.flagged else '✅ SAFE'}"
    )
    
    for result in report.results:
        status = "🚨" if result.flagged else "✅"
        embed.add_field(
            name=f"{status} {result.classifier}",
            value=f"{result.confidence:.1%}",
            inline=True
        )
    
    embed.add_field(name="Text", value=f"```{text[:200]}...```" if len(text) > 200 else f"```{text}```", inline=False)
    embed.set_footer(text=f"Processed in {report.processing_time_ms:.0f}ms")
    
    await interaction.followup.send(embed=embed, ephemeral=True)


@bot.tree.command(name="redact", description="Redact PII from text")
@app_commands.describe(text="The text to redact PII from")
async def redact(interaction: discord.Interaction, text: str):
    """Redact PII from text."""
    if not bot.pii_detector:
        await interaction.response.send_message("❌ PII detector not loaded", ephemeral=True)
        return
    
    redacted, detections = bot.pii_detector.redact(text)
    
    embed = discord.Embed(
        title="🔒 PII Redaction",
        color=discord.Color.blue()
    )
    
    embed.add_field(name="Original", value=f"```{text}```", inline=False)
    embed.add_field(name="Redacted", value=f"```{redacted}```", inline=False)
    
    if detections:
        pii_types = ", ".join(set(d.type for d in detections))
        embed.add_field(name="PII Found", value=pii_types, inline=False)
    else:
        embed.add_field(name="PII Found", value="None", inline=False)
    
    await interaction.response.send_message(embed=embed, ephemeral=True)


@bot.tree.command(name="modhelp", description="Show LocalMod bot help")
async def modhelp(interaction: discord.Interaction):
    """Show help information."""
    embed = discord.Embed(
        title="🛡️ LocalMod Bot Help",
        description="AI-powered content moderation running 100% locally.",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="📋 Commands",
        value=(
            "`/modstats` - View moderation statistics\n"
            "`/analyze <text>` - Analyze text for safety issues\n"
            "`/redact <text>` - Redact PII from text\n"
            "`/modhelp` - Show this help message"
        ),
        inline=False
    )
    
    embed.add_field(
        name="🔍 What I Detect",
        value=(
            "• **Toxicity** - Hate speech, harassment, threats\n"
            "• **Spam** - Promotional content, scams\n"
            "• **Prompt Injection** - LLM jailbreak attempts\n"
            "• **NSFW** - Adult content (text & images)\n"
            "• **PII** - Personal information"
        ),
        inline=False
    )
    
    embed.add_field(
        name="⚙️ Configuration",
        value=f"Log channel: `#{BotConfig.LOG_CHANNEL_NAME}`",
        inline=False
    )
    
    embed.set_footer(text="LocalMod - Your data never leaves your server")
    
    await interaction.response.send_message(embed=embed)


# =============================================================================
# Setup Command (for admins)
# =============================================================================

@bot.tree.command(name="modsetup", description="Setup LocalMod in this server (Admin only)")
@app_commands.default_permissions(administrator=True)
async def modsetup(interaction: discord.Interaction):
    """Setup the bot in a server."""
    await interaction.response.defer(ephemeral=True)
    
    guild = interaction.guild
    
    # Check for existing log channel
    log_channel = discord.utils.get(guild.channels, name=BotConfig.LOG_CHANNEL_NAME)
    
    if not log_channel:
        # Create log channel
        try:
            overwrites = {
                guild.default_role: discord.PermissionOverwrite(read_messages=False),
                guild.me: discord.PermissionOverwrite(read_messages=True, send_messages=True),
            }
            
            log_channel = await guild.create_text_channel(
                BotConfig.LOG_CHANNEL_NAME,
                overwrites=overwrites,
                reason="LocalMod setup - moderation logs"
            )
            
            await log_channel.send(
                embed=discord.Embed(
                    title="🛡️ LocalMod Activated",
                    description="This channel will receive moderation logs.",
                    color=discord.Color.green()
                )
            )
            
            created = True
        except discord.Forbidden:
            await interaction.followup.send("❌ Missing permission to create channels", ephemeral=True)
            return
    else:
        created = False
    
    embed = discord.Embed(
        title="✅ LocalMod Setup Complete",
        color=discord.Color.green()
    )
    
    embed.add_field(
        name="Log Channel",
        value=f"{log_channel.mention} {'(created)' if created else '(existing)'}",
        inline=False
    )
    
    embed.add_field(
        name="Active Classifiers",
        value=", ".join(BotConfig.ENABLED_CLASSIFIERS),
        inline=False
    )
    
    embed.add_field(
        name="Thresholds",
        value=(
            f"• Delete: {BotConfig.DELETE_THRESHOLD:.0%}+\n"
            f"• Timeout: {BotConfig.TIMEOUT_THRESHOLD:.0%}+\n"
        ),
        inline=False
    )
    
    await interaction.followup.send(embed=embed, ephemeral=True)


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the bot."""
    token = os.environ.get("DISCORD_BOT_TOKEN")
    
    if not token:
        print("❌ DISCORD_BOT_TOKEN environment variable not set!")
        print("\nTo run the bot:")
        print("  1. Create a bot at https://discord.com/developers/applications")
        print("  2. Copy the bot token")
        print("  3. Run: export DISCORD_BOT_TOKEN=your_token_here")
        print("  4. Run: python examples/discord_bot.py")
        sys.exit(1)
    
    print("🚀 Starting LocalMod Discord Bot...")
    print("   Classifiers:", BotConfig.ENABLED_CLASSIFIERS)
    print("   Image moderation:", "Enabled" if BotConfig.MODERATE_IMAGES else "Disabled")
    print()
    
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()

