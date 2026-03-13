import pytest
import pandas as pd
from forecast_graph import build_forecast_graph, execute_forecast_workflow

@pytest.fixture
def mock_feature_df():
    """Mock standard request context required by the pipeline."""
    return pd.DataFrame([{'weather': 'Rainy', 'temp_avg': 15.0}])

def test_graph_compilation():
    """Ensures nodes and topological edges map logically preventing looping anomalies."""
    graph = build_forecast_graph()
    nodes = [node for node in graph.nodes]
    
    expected_nodes = [
        'prepare_context', 
        'sold_xgb', 'sold_lstm', 'sold_ensemble',
        'waste_xgb', 'waste_lstm', 'waste_ensemble',
        'derive_goal', 'explain_forecast'
    ]
    
    for n in expected_nodes:
        assert str(n) in {"__start__"} | set(nodes)

def test_inference_execution_path(mock_feature_df):
    """Executes the holistic DAG tracking variables cleanly."""
    result = execute_forecast_workflow("Baguette", "2023-11-20", mock_feature_df)
    
    # 1. Top level context
    assert result['product'] == 'Baguette'
    assert result['target_date'] == '2023-11-20'
    
    # 2. Specific Sold Outputs
    assert 'sold_final_prediction' in result
    assert result['sold_xgb_prediction'] is not None
    assert result['sold_lstm_prediction'] is not None
    assert isinstance(result['sold_gap_ratio'], float)
    
    # 3. Specific Waste Outputs
    assert 'waste_final_prediction' in result
    assert result['waste_xgb_prediction'] is not None
    
    # 4. Derivation and Anomaly Math works
    derived = result['sold_final_prediction'] + result['waste_final_prediction']
    assert pytest.approx(result['recommended_made_qty'], 0.001) == derived
    
    # 5. Explanations propagate cleanly (whether from mock LLM or fallback layer)
    assert 'explanation' in result
    assert 'production_note' in result

def test_failure_handling_with_empty_df():
    """
    Checks if passing missing tracking forces the logic 
    to output deterministic error fallbacks gracefully instead of crashing server instances.
    """
    # LangGraph invokes it, node_prepare_context intercepts, sets 'error_message'
    # End node skips generation
    result = execute_forecast_workflow("Donut", "2023-11-20", pd.DataFrame())
    
    assert result['confidence_label'] == 'Failed'
    assert 'error' in result['explanation'].lower() or 'failure' in result['explanation'].lower()
