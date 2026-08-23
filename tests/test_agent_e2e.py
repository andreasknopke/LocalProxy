import proxy

def test_agent_runner_writes():
    assert hasattr(proxy, '_run_coworker_agent')
    assert callable(proxy._run_coworker_agent)
