"""LangGraph triage agent: LLM clients, MCP client, graph, scripted scenarios.

This package never imports ``yield_triage.approvals``: the agent cannot mint or
verify approval tokens. See ``graph.py`` for the control flow.
"""
