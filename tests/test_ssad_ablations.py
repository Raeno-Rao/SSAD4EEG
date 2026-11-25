import unittest

from scripts.run_ssad_ablations import STEPS, format_ode_tables, interaction_effects, summarize


class AblationTests(unittest.TestCase):
    def test_paired_interaction_uses_subjects(self):
        records = []
        for subject in (1, 2):
            for variant, ba in (("identity_iso", .5), ("filter_iso", .6),
                                ("identity_aniso", .65), ("full", .8)):
                records.append({"dataset": "bciiv2a", "task": "motor_imagery", "test_subject": subject,
                                "variant": variant, "balanced_accuracy": ba})
        effects = interaction_effects(records)
        self.assertEqual(len(effects), 1)
        self.assertAlmostEqual(effects[0]["mean_pp"], 5.)

    def test_deap_task_average_is_paired_by_subject(self):
        records = []
        for task, ba in (("valence", .6), ("arousal", .8)):
            records.append({"dataset": "deap", "task": task, "test_subject": 1,
                            "variant": "full", "steps": 4, "balanced_accuracy": ba})
        row = next(row for row in summarize(records) if row["task"] == "task_average")
        self.assertAlmostEqual(row["balanced_accuracy_mean"], .7)

    def test_complete_ode_sweep_emits_latex_table(self):
        records = [{"dataset": "bciiv2a", "task": "motor_imagery", "variant": "fixed_checkpoint",
                    "test_subject": subject, "steps": steps, "balanced_accuracy": .5,
                    "weighted_f1": .5, "latency_ms_per_sample": float(steps)}
                   for subject in range(1, 10) for steps in STEPS]
        content = format_ode_tables(summarize(records), "ode_inference")
        self.assertIn("20 & 20", content)
        self.assertIn("\\pm", content)


if __name__ == "__main__":
    unittest.main()
