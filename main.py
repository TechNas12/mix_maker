# main.py

from dotenv import load_dotenv

load_dotenv()


def main():
    from src.ui.components.header import print_banner_only
    from src.ui.components.inputs import collect_inputs
    from src.ui.interface import run_pipeline
    from src.provider.llm import LLM

    print_banner_only()

    user_prompt, style_id, n = collect_inputs(
        style_prompts_file=LLM("").style_prompts_file_path
    )

    run_pipeline(
        user_prompt=user_prompt,
        style_id=style_id,
        n=n,
    )


if __name__ == "__main__":
    main()