import proxy

def test_coworker_agent_mode_writes():
    """COWORKER_AGENT_MODE: der Co-Worker-Agent ist im Proxy verdrahtet."""
    assert hasattr(proxy, '_run_coworker_agent')
    assert callable(proxy._run_coworker_agent)
    # Tunnel-Helfer fuer den Client-Tool-Tunnel
    assert callable(proxy._cw_parse_tunnel_id)
    assert callable(proxy._cw_session_new)
    assert callable(proxy._cw_map_tool_calls_out)
