# LLM-Powered Data Analysis Application
# Uses Streamlit, Groq, Pandas, NumPy and Plotly.

import os
import json
import math
import re
import numbers

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from groq import Groq

# Configure the page.
st.set_page_config(
    page_title="LLM Data Analysis Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("📊 LLM-Powered Data Analysis Assistant")
st.write(
    "Upload a dataset, ask a question in plain language, "
    "and let the LLM generate and run an analysis."
)

# Obtain the API key securely.
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    api_key = st.text_input(
        "Groq API key",
        type="password",
        help="Your key is used for this session and is not displayed."
    )

# Upload a dataset.
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        # Read the uploaded file.
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"Loaded: {uploaded_file.name}")
        st.write(
            f"Rows: {df.shape[0]:,} | "
            f"Columns: {df.shape[1]}"
        )

        
        
        # Display an organised preview of the first five rows.
        with st.expander("Preview dataset", expanded=True):
            preview_df = df.head(5).copy()
            st.table(preview_df)


        # Summarize the dataset for the LLM.
        dataset_info = {
            "number_of_rows": int(df.shape[0]),
            "number_of_columns": int(df.shape[1]),
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }

        question = st.text_input(
            "What would you like to know about your dataset?",
            placeholder="e.g. What is the average monthly charge?"
        )

        # Start the analysis when the user clicks the button.
        if st.button("Analyze dataset", type="primary"):
            if not api_key:
                st.error("Enter your Groq API key to continue.")
            elif not question.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    client = Groq(api_key=api_key)

                    prompt = f"""
You are a data analysis assistant.

User question:
{question}

Dataset information:
{json.dumps(dataset_info, default=str)}

Generate an analysis plan and Python code.

Requirements:
- The DataFrame is named df.
- Use pd, np, and px when needed.
- Do not modify the original DataFrame.
- Store the answer in result.
- If useful, store a Plotly figure in fig.
- Use only the provided DataFrame and libraries.
- Do not use file access, network access, os, subprocess,
  eval, exec, or imports in generated code.
- List columns used.
- Return only a valid JSON object with:
  analysis_plan, python_code, columns_used.
"""

                    with st.spinner("Generating analysis plan and code..."):
                        response = client.chat.completions.create(
                            model="openai/gpt-oss-20b",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "Return valid JSON only. "
                                        "Generate concise, relevant analysis "
                                        "code for the supplied dataset."
                                    )
                                },
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0,
                            max_completion_tokens=3000
                        )

                    raw_response = response.choices[0].message.content

                    if not raw_response or not raw_response.strip():
                        raise ValueError("The LLM returned an empty response.")

                    # Parse the JSON response.
                    cleaned = raw_response.strip()
                    cleaned = re.sub(
                        r"^```(?:json)?\s*", "", cleaned
                    )
                    cleaned = re.sub(r"\s*```$", "", cleaned)

                    start = cleaned.find("{")
                    end = cleaned.rfind("}")

                    if start == -1 or end == -1:
                        raise ValueError("No JSON object was returned.")

                    analysis = json.loads(cleaned[start:end + 1])

                    for field in (
                        "analysis_plan", "python_code", "columns_used"
                    ):
                        if field not in analysis:
                            raise ValueError(
                                f"LLM response is missing {field}."
                            )

                    # Validate referenced columns.
                    missing = [
                        c for c in analysis["columns_used"]
                        if c not in df.columns
                    ]
                    if missing:
                        raise ValueError(
                            f"Unknown dataset columns: {missing}"
                        )

                    # Decode escaped line breaks from JSON code strings.
                    code = analysis["python_code"]
                    code = code.replace("\\n", "\n")

                    # Basic code restrictions.
                    forbidden = [
                        "import ", "open(", "exec(", "eval(",
                        "__import__", "subprocess", "requests",
                        "urllib", "os.", "globals(", "locals("
                    ]

                    if any(term in code for term in forbidden):
                        raise ValueError(
                            "Generated code contains a restricted operation. "
                            "Please rephrase your question and try again."
                        )

                    # Execute in a restricted namespace.
                    # This is a basic demonstration, not a secure sandbox.
                    namespace = {
                        "__builtins__": {
                            "len": len,
                            "sum": sum,
                            "min": min,
                            "max": max,
                            "abs": abs,
                            "round": round,
                            "float": float,
                            "int": int,
                            "str": str,
                            "list": list,
                            "dict": dict,
                            "tuple": tuple,
                            "set": set,
                            "range": range,
                            "enumerate": enumerate,
                            "zip": zip,
                            "sorted": sorted,
                            "print": print,
                            "ValueError": ValueError
                        },
                        "pd": pd,
                        "np": np,
                        "px": px,
                        "df": df.copy()
                    }

                    with st.spinner("Executing analysis..."):
                        exec(code, namespace)

                    if "result" not in namespace:
                        raise ValueError(
                            "The generated code did not produce result."
                        )

                    result = namespace["result"]
                    fig = namespace.get("fig")

                    # Display the analysis plan.
                    st.subheader("Analysis plan")
                    st.write(analysis["analysis_plan"])

                    # Display the generated code.
                    with st.expander("View generated Python code"):
                        st.code(code, language="python")

                    # Display the result.
                    st.subheader("Analysis result")

                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result, use_container_width=True)
                    elif isinstance(result, pd.Series):
                        st.dataframe(
                            result.to_frame(),
                            use_container_width=True
                        )
                    elif isinstance(result, numbers.Number):
                        if not math.isfinite(float(result)):
                            raise ValueError(
                                "The result is not a finite number."
                            )
                        st.metric("Calculated result", f"{result:,.2f}")
                    else:
                        st.write(result)

                    # Display the visualization if one exists.
                    if fig is not None:
                        st.subheader("Visualization")
                        st.plotly_chart(fig, use_container_width=True)

                    # Prepare a concise representation for the explanation.
                    if isinstance(result, pd.DataFrame):
                        result_summary = result.head(10).to_dict(
                            orient="records"
                        )
                    elif isinstance(result, pd.Series):
                        result_summary = result.head(10).to_dict()
                    else:
                        result_summary = str(result)

                    # Generate a plain-language explanation.
                    explanation_prompt = f"""
Explain this data analysis in plain language.

Question: {question}
Analysis plan: {analysis['analysis_plan']}
Result: {json.dumps(result_summary, default=str)}
Dataset: {uploaded_file.name}
Columns used: {analysis['columns_used']}

Do not invent findings or claim causation.
"""

                    with st.spinner("Generating explanation..."):
                        explanation_response = (
                            client.chat.completions.create(
                                model="openai/gpt-oss-20b",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": (
                                            "Explain results clearly and "
                                            "accurately in plain language."
                                        )
                                    },
                                    {
                                        "role": "user",
                                        "content": explanation_prompt
                                    }
                                ],
                                temperature=0,
                                max_completion_tokens=500
                            )
                        )

                    explanation = (
                        explanation_response.choices[0].message.content
                    )

                    st.subheader("Plain-language explanation")
                    if explanation and explanation.strip():
                        st.write(explanation)
                    else:
                        st.warning(
                            "The LLM returned an empty explanation."
                        )

                    # Display source references.
                    st.subheader("Data and source references")
                    st.write(f"**Source dataset:** {uploaded_file.name}")
                    st.write(f"**Rows:** {df.shape[0]:,}")
                    st.write(f"**Columns:** {df.shape[1]}")
                    st.write(f"**Question:** {question}")
                    st.write(
                        "**Columns used:** "
                        + ", ".join(analysis["columns_used"])
                    )
                    st.write("**LLM provider:** Groq")
                    st.write("**Model:** openai/gpt-oss-20b")

                except Exception as e:
                    st.error(
                        f"Analysis failed: {type(e).__name__}: {e}"
                    )
                    st.info(
                        "Check the question, dataset columns, API key, "
                        "and API availability, then try again."
                    )

    except Exception as e:
        st.error(
            f"Could not read the uploaded dataset: "
            f"{type(e).__name__}: {e}"
        )
else:
    st.info("Upload a CSV or Excel file to get started.")
