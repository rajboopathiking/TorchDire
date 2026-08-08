import os
import tempfile
import torch
from torchdire.profiler.efficiency import QGFDProfiler, profile_qgfd_efficiency
from torchdire.experiments.ablation import QGFDAblator, SmallModelForAblation


def test_qgfd_profiler():
    profiler = QGFDProfiler(device="cpu")
    results = profiler.profile_layer(
        batch_size=2,
        seq_len=64,
        embed_dim=64,
        num_heads=4,
        diffusion_steps=2,
        target_alpha=0.02,
        mode="full",
        num_runs=2,
    )
    assert results["batch_size"] == 2
    assert results["seq_len"] == 64
    assert results["baseline_latency_ms"] > 0
    assert results["qgfd_latency_ms"] > 0
    assert results["estimated_vram_mb"] > 0

    results_conv = profile_qgfd_efficiency(
        batch_size=2,
        seq_len=64,
        embed_dim=64,
        mode="conv",
    )
    assert results_conv["mode"] == "conv"


def test_qgfd_ablator():
    ablator = QGFDAblator(
        steps_list=[2],
        alpha_list=[0.02],
        detach_p_list=[True],
        warmup_list=[2000],
        device="cpu",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "ablation.csv")
        results = ablator.run(save_csv_path=csv_path)

        assert len(results) == 1
        assert results[0]["Steps"] == 2
        assert results[0]["Alpha"] == 0.02
        assert os.path.exists(csv_path)


def test_small_ablation_model_forward():
    model = SmallModelForAblation(vocab_size=100, embed_dim=64, num_heads=4, diffusion_steps=2)
    inputs = torch.randint(0, 100, (2, 16))
    logits = model(inputs)
    assert logits.shape == (2, 16, 100)
    assert not torch.isnan(logits).any()


if __name__ == "__main__":
    test_qgfd_profiler()
    test_qgfd_ablator()
    test_small_ablation_model_forward()
    print("All profiler and experiment tests passed successfully!")
