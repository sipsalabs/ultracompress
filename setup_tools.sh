#!/bin/bash
# Setup MCP tools for Claude Code — run once
# This gives Claude persistent memory + Obsidian integration

echo "Setting up MCP tools for Claude Code..."

# 1. Persistent memory (knowledge graph across all sessions)
echo "Adding memory MCP server..."
claude mcp add memory -- npx @modelcontextprotocol/server-memory

# 2. Web fetching (read web pages)
echo "Adding fetch MCP server..."
claude mcp add fetch -- npx @modelcontextprotocol/server-fetch

# 3. Obsidian integration (if you have an Obsidian vault)
# Uncomment and set your vault path:
# VAULT_PATH="$HOME/Documents/Obsidian"
# echo "Adding Obsidian MCP server..."
# claude mcp add obsidian -- npx obsidian-mcp "$VAULT_PATH"

echo ""
echo "Done! Restart Claude Code for changes to take effect."
echo "Claude will now have persistent memory across all sessions."
