
    explorer_df = filtered_df.copy()

    if search_text.strip():
        search_mask = pd.Series(
            False,
            index=explorer_df.index,
        )

        for column in [
            "Facility",
            "Diagnosis",
            "County",
            "Record_ID",
            "Region",
        ]:
            search_mask |= explorer_df[column].astype(str).str.contains(
                search_text.strip(),
                case=False,
                na=False,
                regex=False,
            )

        explorer_df = explorer_df[search_mask]

    if explorer_df.empty:
        st.info("No records match the current search.")
    else:
        minimum_los = float(
            max(0, np.floor(explorer_df["LOS_Days"].min()))
        )

        maximum_los = float(
            np.ceil(explorer_df["LOS_Days"].max())
        )

        if maximum_los <= minimum_los:
            maximum_los = minimum_los + 1

        los_range = st.slider(
            "Length-of-stay range (days)",
            min_value=minimum_los,
            max_value=maximum_los,
            value=(minimum_los, maximum_los),
        )

        explorer_df = explorer_df[
            explorer_df["LOS_Days"].between(
                los_range[0],
                los_range[1],
            )
        ]

        if explorer_df.empty:
            st.info("No records fall within the selected stay range.")
        else:
            charge_maximum = float(
                explorer_df["Total_Charges"].max()
            )

            charge_maximum = max(charge_maximum, 1.0)

            charge_limit = st.number_input(
                "Maximum total charges ($)",
                min_value=0.0,
                max_value=charge_maximum,
                value=charge_maximum,
                step=max(charge_maximum / 20, 1.0),
            )

            explorer_df = explorer_df[
                explorer_df["Total_Charges"] <= charge_limit
            ]

            st.metric(
                "Records matching explorer filters",
                format_number(len(explorer_df)),
            )

            st.dataframe(
                explorer_df,
                use_container_width=True,
                hide_index=True,
                height=500,
            )

            st.download_button(
                "Download displayed records as CSV",
                data=explorer_df.to_csv(index=False).encode("utf-8"),
                file_name="mediscope_filtered_records.csv",
                mime="text/csv",
                use_container_width=True,
            )

            numeric_columns = [
                column for column in [
                    "LOS_Days",
                    "Total_Charges",
                    "Discharge_Year",
                ]
                if column in explorer_df.columns
                and pd.api.types.is_numeric_dtype(explorer_df[column])
            ]

            if numeric_columns:
                st.subheader("Numeric summary")

                st.dataframe(
                    explorer_df[numeric_columns].describe().transpose(),
                    use_container_width=True,
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong style="color:#00e5ff;">MediScope Health Analytics</strong><br>
        Hospital Inpatient Discharge Intelligence Dashboard<br>
        Built with Streamlit, Pandas, Plotly, and scikit-learn<br>
        <span style="color:#526b85;">
            Analytics are descriptive and educational. They should not be
            interpreted as clinical advice or proof of hospital quality.
        </span>
    </div>
