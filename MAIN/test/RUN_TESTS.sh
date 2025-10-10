#!/bin/bash
# Quick test script for MAIN IK Pipeline

set -e

echo "========================================="
echo "MAIN IK Pipeline - Quick Test Suite"
echo "========================================="
echo ""

# Test 1: Example Workflow
echo "Test 1: Running example workflow..."
uv run python example_workflow.py > /dev/null 2>&1
if [ -f example_results.npz ]; then
    echo "✅ Example workflow: PASSED"
else
    echo "❌ Example workflow: FAILED"
fi
echo ""

# Test 2: Orientation IK
echo "Test 2: Testing orientation IK (single frame)..."
uv run python ik_run_orients.py --mode test --frame 0 --n-iter 30 --output test_orient.npz --verbosity 0 > /dev/null 2>&1
if [ -f test_orient.npz ]; then
    echo "✅ Orientation IK: PASSED"
else
    echo "❌ Orientation IK: FAILED"
fi
echo ""

# Test 3: Visualization
echo "Test 3: Testing visualization (PNG)..."
uv run python visualize_animation.py --input test_orient.npz --output test_viz.png --format png > /dev/null 2>&1
if [ -d test_viz.png ]; then
    echo "✅ Visualization: PASSED"
else
    echo "❌ Visualization: FAILED"
fi
echo ""

# Test 4: Marker Editor
echo "Test 4: Testing marker editor (console)..."
echo -e "add 100 test\nsave quick_test.npz\nquit" | uv run python marker_editor.py --console-only > /dev/null 2>&1
if [ -f quick_test.npz ]; then
    echo "✅ Marker editor: PASSED"
else
    echo "❌ Marker editor: FAILED"
fi
echo ""

echo "========================================="
echo "Test suite complete!"
echo "========================================="
echo ""
echo "Output files:"
ls -lh *.npz *.gif test_viz.png/*.png 2>/dev/null | tail -10
echo ""
echo "To view animation:"
echo "  uv run python visualize_animation.py --input example_results.npz --output anim.gif --fps 10"
