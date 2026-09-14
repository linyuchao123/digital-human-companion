import unittest

from services.agent import EmotionAnalyzer, RiskLevel, SafetyDecision


class EmotionAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = EmotionAnalyzer()

    def test_detects_anxiety_with_valence_and_arousal(self):
        result = self.analyzer.analyze("最近压力很大，总是睡不着", SafetyDecision())

        self.assertEqual(result.emotion, "Anxiety")
        self.assertLess(result.valence, 0)
        self.assertGreater(result.arousal, 0)

    def test_high_risk_safety_decision_has_priority(self):
        result = self.analyzer.analyze(
            "我没事",
            SafetyDecision(risk_level=RiskLevel.HIGH, requires_safe_response=True),
        )

        self.assertEqual(result.emotion, "Concerned")
        self.assertEqual(result.label, "紧急关切")

    def test_defaults_to_neutral(self):
        result = self.analyzer.analyze("今天吃了午饭", SafetyDecision())

        self.assertEqual(result.emotion, "Neutral")
        self.assertEqual(result.label, "平静")


if __name__ == "__main__":
    unittest.main()
