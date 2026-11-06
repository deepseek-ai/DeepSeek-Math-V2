import pytest
import sys
from pathlib import Path

# Add inference directory to sys.path
inference_dir = Path(__file__).resolve().parent.parent / "inference"
sys.path.insert(0, str(inference_dir))

from main import get_parser as get_main_parser
from generate import get_parser as get_generate_parser


class TestMainArgumentParsing:
    def test_default_endpoint_configuration(self):
        """Verify default values when endpoints are not passed."""
        parser = get_main_parser()
        args, _ = parser.parse_known_args([
            "--input_paths", "inputs/IMO2025.json",
            "--output_dirname", "outputs/test",
            "--proof_pool_dirname", "outputs/test/proof_pool",
        ])
        assert args.proof_gen_url is None
        assert args.proof_rate_url is None
        assert args.infer_script == "generate"
        assert args.input_paths == "inputs/IMO2025.json"
        assert args.output_dirname == "outputs/test"
        assert args.proof_pool_dirname == "outputs/test/proof_pool"
        assert args.batch_size == 160
        assert args.start_round == 1
        assert args.max_rounds == 20

    def test_explicit_endpoint_configuration(self):
        """Verify explicit endpoint URLs and infer_script are stored correctly."""
        parser = get_main_parser()
        args, _ = parser.parse_known_args([
            "--input_paths", "inputs/IMO2025.json",
            "--output_dirname", "outputs/test",
            "--proof_pool_dirname", "outputs/test/proof_pool",
            "--proof_gen_url", "https://api.generator.example.com/v1",
            "--proof_rate_url", "https://api.verifier.example.com/v1",
            "--infer_script", "custom_generate",
        ])
        assert args.proof_gen_url == "https://api.generator.example.com/v1"
        assert args.proof_rate_url == "https://api.verifier.example.com/v1"
        assert args.infer_script == "custom_generate"

    def test_missing_required_configuration(self):
        """Verify parser raises SystemExit when required arguments are missing."""
        parser = get_main_parser()

        # Missing all required arguments
        with pytest.raises(SystemExit):
            parser.parse_args([])

        # Missing proof_pool_dirname
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--input_paths", "inputs/IMO2025.json",
                "--output_dirname", "outputs/test",
            ])

        # Missing input_paths
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--output_dirname", "outputs/test",
                "--proof_pool_dirname", "outputs/test/proof_pool",
            ])

    def test_unknown_command_line_arguments(self):
        """Verify parse_known_args retains unrecognized arguments without error."""
        parser = get_main_parser()
        args, unknown = parser.parse_known_args([
            "--input_paths", "inputs/IMO2025.json",
            "--output_dirname", "outputs/test",
            "--proof_pool_dirname", "outputs/test/proof_pool",
            "--custom_extra_flag", "extra_val",
        ])
        assert args.input_paths == "inputs/IMO2025.json"
        assert "--custom_extra_flag" in unknown
        assert "extra_val" in unknown


class TestGenerateArgumentParsing:
    def test_default_api_url(self):
        """Verify generate.py accepts args and defaults --api_url to None."""
        parser = get_generate_parser()
        args, _ = parser.parse_known_args([
            "--input_data_path", "in.jsonl",
            "--output_data_path", "out.jsonl",
            "--temperature", "1.0",
            "--top_p", "0.95",
            "--max_tokens", "1024",
            "--n", "1",
        ])
        assert args.api_url is None
        assert args.input_data_path == "in.jsonl"
        assert args.output_data_path == "out.jsonl"
        assert args.num_processes == 16
        assert args.batch_size == 16

    def test_explicit_api_url(self):
        """Verify generate.py stores explicit --api_url."""
        parser = get_generate_parser()
        args, _ = parser.parse_known_args([
            "--input_data_path", "in.jsonl",
            "--output_data_path", "out.jsonl",
            "--api_url", "https://api.openai-compatible.example.com/v1",
            "--temperature", "0.7",
            "--top_p", "0.9",
            "--max_tokens", "2048",
            "--n", "4",
        ])
        assert args.api_url == "https://api.openai-compatible.example.com/v1"
        assert args.temperature == 0.7
        assert args.n == 4
