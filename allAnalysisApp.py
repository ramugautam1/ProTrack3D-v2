import numpy as np
import pandas as pd
from functions import smooth
import os
import tifffile
import nibabel as nib
import matplotlib.pyplot as plt
import json
import sqlite3
from matplotlib.colors import LinearSegmentedColormap
from itertools import zip_longest


import sqlite3
import pandas as pd
import re

# ------------------------------------------------------------------
# Quartiles from size within [t1, t2]
# ------------------------------------------------------------------
def _compute_size_q123(conn, t1, t2):
    df = pd.read_sql_query(
        "SELECT size FROM object_properties WHERE t>=? AND t<=?",
        conn,
        params=(int(t1), int(t2)),
    )
    if df.empty:
        return 0, 0, 0
    return (
        int(df["size"].quantile(0.25)),
        int(df["size"].quantile(0.50)),
        int(df["size"].quantile(0.75)),
    )


# ------------------------------------------------------------------
# Create TEMP VIEW with dynamic sizeQ / previous_sizeQ / next_sizeQ
# ------------------------------------------------------------------
def _ensure_dyn_size_view(conn, t1, t2, view="object_properties_dyn"):
    q1, q2, q3 = _compute_size_q123(conn, t1, t2)

    def qcase(expr):
        return f"""
        (CASE
            WHEN ({expr}) <= {q1} THEN 1
            WHEN ({expr}) <= {q2} THEN 2
            WHEN ({expr}) <= {q3} THEN 3
            ELSE 4
        END)
        """

    sizeQ_expr = qcase("op.size")

    # Only compute prev/next quartiles when valid
    prevQ_expr = f"""
    (CASE
        WHEN op.existed_previous THEN
            {qcase("op.size - COALESCE(op.size_change_from_prev,0)")}
        ELSE NULL
    END)
    """

    nextQ_expr = f"""
    (CASE
        WHEN op.exists_next THEN
            {qcase("op.size + COALESCE(op.size_change_next,0)")}
        ELSE NULL
    END)
    """

    cur = conn.cursor()
    cur.execute(f"DROP VIEW IF EXISTS {view}")

    cur.execute(f"""
    CREATE TEMP VIEW {view} AS
    SELECT
        op.t,
        op.id,
        op.size,
        op.intensity,
        op.peak_intensity,
        op.density,

        {sizeQ_expr} AS sizeQ,

        op.intensityQ,
        op.remaining_life,

        {prevQ_expr} AS previous_sizeQ,
        op.previous_intensityQ,

        {nextQ_expr} AS next_sizeQ,
        op.next_intensityQ,

        op.size_change_next,
        op.size_change_from_prev,
        op.intensity_change_next,
        op.intensity_change_from_prev,

        op.born_at_t,
        op.dead_after_t,
        op.splitted_at_t,
        op.split_off_at_t,
        op.merge_primary_at_t,
        op.merge_primary_at_next_t,
        op.merge_secondary_at_next_t,
        op.will_split_next,
        op.existed_previous,
        op.exists_next
    FROM object_properties op
    WHERE op.t >= {int(t1)} AND op.t <= {int(t2)};
    """)

    return view, (q1, q2, q3)


# ------------------------------------------------------------------
# Rewrite SQL so queries transparently use the view
# ------------------------------------------------------------------
def _rewrite_sql_to_view(sql, view):
    sql2 = re.sub(
        r"(?i)\bfrom\s+object_properties\b",
        f"FROM {view}",
        sql,
    )
    return sql2


# ------------------------------------------------------------------
# PUBLIC DROP-IN FUNCTION (same name you already use)
# ------------------------------------------------------------------
def dynq_execute(conn, t1, t2, sql, debug=False):
    """
    Drop-in replacement:
      columns, results = dynq_execute(conn, t1, t2, query)

    Internally:
      - creates TEMP VIEW limited to [t1,t2]
      - recomputes sizeQ / previous_sizeQ / next_sizeQ
      - preserves NULL prev/next when object did not exist
      - rewrites SQL to use the view
    """

    view, (q1, q2, q3) = _ensure_dyn_size_view(conn, t1, t2)

    sql2 = _rewrite_sql_to_view(sql, view)

    if debug:
        print(f"[dynq] t1={t1}, t2={t2}, q1={q1}, q2={q2}, q3={q3}")
        print("[dynq] SQL head:", sql2[:300].replace("\n", " "))

    cur = conn.cursor()
    cur.execute(sql2)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    return cols, rows



colors=['blue','purple','magenta','red']
color5 = ['green']+ colors
color6 = ['orange','green','purple','magenta','cyan','red']

category = 'size'

def merge_dataframes_and_save(df_list, output_file):
    dflist=[]
    for df in df_list:
        if 't' in df.columns:
            # df = df.dropna()
            # print(df.head())
            df['t'] = df['t'].astype('Int64')  # Use 'Int64' for nullable integers
            mint = df['t'].min()
            maxt = df['t'].max()
            complete_range = pd.DataFrame({'t': range(int(mint), int(maxt + 1))})
            df_filled = pd.merge(complete_range, df, on='t', how='left')
            df_filled = df_filled.fillna(0)
            dflist.append(df_filled)
        else:
            dflist.append(df)

    # Merge the DataFrames
    merged_df = pd.concat(dflist, axis=1)

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(output_file, index=False)

    print(f"Merged DataFrame saved to {output_file}")
    return merged_df

###########################################

def getEventIntensityPlots(plotsavepath, sT, trackedimagepath, origImgPath, t1=None, t2=None):
    dbpath = os.path.dirname(trackedimagepath)
    samplename = os.path.basename(origImgPath)[:-4]
    #######################################################################

    query = f"""
        SELECT
            t,
            -- Counts for sizeQ = 1 (Q1)
            SUM(CASE WHEN sizeQ = 1 AND next_sizeQ > sizeQ THEN 1 ELSE 0 END) AS sizeInc_Q1,
            SUM(CASE WHEN sizeQ = 1 AND next_sizeQ = sizeQ THEN 1 ELSE 0 END) AS sizeEq_Q1,
            SUM(CASE WHEN sizeQ = 1 AND next_sizeQ < sizeQ THEN 1 ELSE 0 END) AS sizeDec_Q1,
            COUNT(CASE WHEN sizeQ = 1 THEN id END) AS count_Q1,  -- Count for sizeQ = 1

            -- Counts for sizeQ = 2 (Q2)
            SUM(CASE WHEN sizeQ = 2 AND next_sizeQ > sizeQ THEN 1 ELSE 0 END) AS sizeInc_Q2,
            SUM(CASE WHEN sizeQ = 2 AND next_sizeQ = sizeQ THEN 1 ELSE 0 END) AS sizeEq_Q2,
            SUM(CASE WHEN sizeQ = 2 AND next_sizeQ < sizeQ THEN 1 ELSE 0 END) AS sizeDec_Q2,
            COUNT(CASE WHEN sizeQ = 2 THEN id END) AS count_Q2,  -- Count for sizeQ = 2

            -- Counts for sizeQ = 3 (Q3)
            SUM(CASE WHEN sizeQ = 3 AND next_sizeQ > sizeQ THEN 1 ELSE 0 END) AS sizeInc_Q3,
            SUM(CASE WHEN sizeQ = 3 AND next_sizeQ = sizeQ THEN 1 ELSE 0 END) AS sizeEq_Q3,
            SUM(CASE WHEN sizeQ = 3 AND next_sizeQ < sizeQ THEN 1 ELSE 0 END) AS sizeDec_Q3,
            COUNT(CASE WHEN sizeQ = 3 THEN id END) AS count_Q3,  -- Count for sizeQ = 3

            -- Counts for sizeQ = 4 (Q4)
            SUM(CASE WHEN sizeQ = 4 AND next_sizeQ > sizeQ THEN 1 ELSE 0 END) AS sizeInc_Q4,
            SUM(CASE WHEN sizeQ = 4 AND next_sizeQ = sizeQ THEN 1 ELSE 0 END) AS sizeEq_Q4,
            SUM(CASE WHEN sizeQ = 4 AND next_sizeQ < sizeQ THEN 1 ELSE 0 END) AS sizeDec_Q4,
            COUNT(CASE WHEN sizeQ = 4 THEN id END) AS count_Q4   -- Count for sizeQ = 4

        FROM
            object_properties
        WHERE
            exists_next
        GROUP BY
            t
        ORDER BY
            t;
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)
    FlowObj = newDF[(newDF['t'] > t1) & (newDF['t'] <= t2)]

    # Calculate fractions for sizeQ1
    FlowObj['Inc_Q1'] = np.where(FlowObj['count_Q1'] > 0, FlowObj['sizeInc_Q1'] / FlowObj['count_Q1'], 0)
    FlowObj['Eq_Q1'] = np.where(FlowObj['count_Q1'] > 0, FlowObj['sizeEq_Q1'] / FlowObj['count_Q1'], 0)
    FlowObj['Dec_Q1'] = np.where(FlowObj['count_Q1'] > 0, FlowObj['sizeDec_Q1'] / FlowObj['count_Q1'], 0)

    # Calculate fractions for sizeQ2
    FlowObj['Inc_Q2'] = np.where(FlowObj['count_Q2'] > 0, FlowObj['sizeInc_Q2'] / FlowObj['count_Q2'], 0)
    FlowObj['Eq_Q2'] = np.where(FlowObj['count_Q2'] > 0, FlowObj['sizeEq_Q2'] / FlowObj['count_Q2'], 0)
    FlowObj['Dec_Q2'] = np.where(FlowObj['count_Q2'] > 0, FlowObj['sizeDec_Q2'] / FlowObj['count_Q2'], 0)

    # Calculate fractions for sizeQ3
    FlowObj['Inc_Q3'] = np.where(FlowObj['count_Q3'] > 0, FlowObj['sizeInc_Q3'] / FlowObj['count_Q3'], 0)
    FlowObj['Eq_Q3'] = np.where(FlowObj['count_Q3'] > 0, FlowObj['sizeEq_Q3'] / FlowObj['count_Q3'], 0)
    FlowObj['Dec_Q3'] = np.where(FlowObj['count_Q3'] > 0, FlowObj['sizeDec_Q3'] / FlowObj['count_Q3'], 0)

    # Calculate fractions for sizeQ4
    FlowObj['Inc_Q4'] = np.where(FlowObj['count_Q4'] > 0, FlowObj['sizeInc_Q4'] / FlowObj['count_Q4'], 0)
    FlowObj['Eq_Q4'] = np.where(FlowObj['count_Q4'] > 0, FlowObj['sizeEq_Q4'] / FlowObj['count_Q4'], 0)
    FlowObj['Dec_Q4'] = np.where(FlowObj['count_Q4'] > 0, FlowObj['sizeDec_Q4'] / FlowObj['count_Q4'], 0)

    # Create separate DataFrames for each sizeQ category
    sizeQ1_df = FlowObj[['t', 'Inc_Q1', 'Eq_Q1', 'Dec_Q1']]
    sizeQ2_df = FlowObj[['t', 'Inc_Q2', 'Eq_Q2', 'Dec_Q2']]
    sizeQ3_df = FlowObj[['t', 'Inc_Q3', 'Eq_Q3', 'Dec_Q3']]
    sizeQ4_df = FlowObj[['t', 'Inc_Q4', 'Eq_Q4', 'Dec_Q4']]

    # Optionally, reset index if needed
    sizeQ1_df.reset_index(drop=True, inplace=True)
    sizeQ2_df.reset_index(drop=True, inplace=True)
    sizeQ3_df.reset_index(drop=True, inplace=True)
    sizeQ4_df.reset_index(drop=True, inplace=True)

    flowDFs = [sizeQ1_df, sizeQ2_df, sizeQ3_df, sizeQ4_df]

    titles = ['Q1', 'Q2', 'Q3', 'Q4']

    y_axs = ['Fraction of Objects',
             'Fraction of Objects',
             'Fraction of Objects',
             'Fraction of Objects',
             'Fraction of Objects',
             ]

    ix = 4
    jx = 1
    fig, axes = plt.subplots(ix, jx, figsize=(10 * jx, ix * 10))
    for i in range(ix):
        ax = axes[i]
        df = flowDFs[i].drop(columns=['t'])
        smooth(df).plot(kind='line', ax=ax,
                                                         color=['green', 'orange', 'red'])
        #                 ax.axvline(x=acp, color='green', linestyle='--', label='apical constriction point')
        ax.set_title(titles[i] , fontsize=15)
        ax.set_xlabel('Timepoint', fontsize=15)
        ax.set_ylabel(y_axs[i], fontsize=15)

    plt.tight_layout()
    plt.savefig(plotsavepath + '/ObjectFlowDiagram.png')
    plt.close()
    #######################################################################
    #######################################################################
    #######################################################################
    #######################################################################
    #######################################################################
    #######################################################################
    # THE FOLLOWING SECTION OF CODE PRODUCES THE BIG GRAPH WITH MANY PLOTS
    #######################################################################
    #######################################################################
    #######################################################################
    #######################################################################
    #######################################################################
    #######################################################################
    #######################################################################

    query = f"""
             SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN 1 END, 0)) AS Q1Count,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN 1 END, 0)) AS Q2Count,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN 1 END, 0)) AS Q3Count,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN 1 END, 0)) AS Q4Count
                FROM
                    object_properties
                GROUP BY
                    t

    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    totObj = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]
    totObj

    #######################################################################

    event = 'split'

    query = f"""
        SELECT
            t1.t,
            CASE WHEN t2.AllQ1Count <> 0 THEN t1.{event}Q1Count * 1.0 / t2.AllQ1Count ELSE 0 END AS Q1{event}Frac,
            CASE WHEN t2.AllQ2Count <> 0 THEN t1.{event}Q2Count * 1.0 / t2.AllQ2Count ELSE 0 END AS Q2{event}Frac,
            CASE WHEN t2.AllQ3Count <> 0 THEN t1.{event}Q3Count * 1.0 / t2.AllQ3Count ELSE 0 END AS Q3{event}Frac,
            CASE WHEN t2.AllQ4Count <> 0 THEN t1.{event}Q4Count * 1.0 / t2.AllQ4Count ELSE 0 END AS Q4{event}Frac
        FROM
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS {event}Q1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS {event}Q2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS {event}Q3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                will_split_next and merge_primary_at_next_t=0
            GROUP BY
                t) AS t1
        INNER JOIN
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS AllQ1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS AllQ2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS AllQ3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS AllQ4Count
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    splitObj = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    splitObj

    print('---'*20, splitObj.columns)


    #######################################################################


    event = 'Merge'

    query = f"""
        SELECT
            t1.t,
            CASE WHEN t2.AllQ1Count <> 0 THEN t1.{event}Q1Count * 1.0 / t2.AllQ1Count ELSE 0 END AS Q1{event}Frac,
            CASE WHEN t2.AllQ2Count <> 0 THEN t1.{event}Q2Count * 1.0 / t2.AllQ2Count ELSE 0 END AS Q2{event}Frac,
            CASE WHEN t2.AllQ3Count <> 0 THEN t1.{event}Q3Count * 1.0 / t2.AllQ3Count ELSE 0 END AS Q3{event}Frac,
            CASE WHEN t2.AllQ4Count <> 0 THEN t1.{event}Q4Count * 1.0 / t2.AllQ4Count ELSE 0 END AS Q4{event}Frac
        FROM
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS {event}Q1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS {event}Q2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS {event}Q3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                (merge_primary_at_next_t or merge_secondary_at_next_t)
            GROUP BY
                t) AS t1
        INNER JOIN
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS AllQ1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS AllQ2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS AllQ3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS AllQ4Count
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    mergeObj = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    mergeObj

    #######################################################################
    #######################################################################

    event = 'mergePrimary'

    query = f"""
        SELECT
            t1.t,
            CASE WHEN t2.AllQ1Count <> 0 THEN t1.{event}Q1Count * 1.0 / t2.AllQ1Count ELSE 0 END AS Q1{event}Frac,
            CASE WHEN t2.AllQ2Count <> 0 THEN t1.{event}Q2Count * 1.0 / t2.AllQ2Count ELSE 0 END AS Q2{event}Frac,
            CASE WHEN t2.AllQ3Count <> 0 THEN t1.{event}Q3Count * 1.0 / t2.AllQ3Count ELSE 0 END AS Q3{event}Frac,
            CASE WHEN t2.AllQ4Count <> 0 THEN t1.{event}Q4Count * 1.0 / t2.AllQ4Count ELSE 0 END AS Q4{event}Frac
        FROM
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS {event}Q1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS {event}Q2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS {event}Q3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                (merge_primary_at_next_t  )
            GROUP BY
                t) AS t1
        INNER JOIN
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS AllQ1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS AllQ2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS AllQ3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS AllQ4Count
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    mergePrimObj = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    # mergeObj.rolling(window=21, min_periods=1).mean().plot(color = color5, figsize=(10,6), title=f'Merge')
    mergePrimObj

    #######################################################################

    event = 'mergeSecondary'

    query = f"""
        SELECT
            t1.t,
            CASE WHEN t2.AllQ1Count <> 0 THEN t1.{event}Q1Count * 1.0 / t2.AllQ1Count ELSE 0 END AS Q1{event}Frac,
            CASE WHEN t2.AllQ2Count <> 0 THEN t1.{event}Q2Count * 1.0 / t2.AllQ2Count ELSE 0 END AS Q2{event}Frac,
            CASE WHEN t2.AllQ3Count <> 0 THEN t1.{event}Q3Count * 1.0 / t2.AllQ3Count ELSE 0 END AS Q3{event}Frac,
            CASE WHEN t2.AllQ4Count <> 0 THEN t1.{event}Q4Count * 1.0 / t2.AllQ4Count ELSE 0 END AS Q4{event}Frac
        FROM
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS {event}Q1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS {event}Q2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS {event}Q3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                (merge_secondary_at_next_t  )
            GROUP BY
                t) AS t1
        INNER JOIN
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS AllQ1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS AllQ2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS AllQ3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS AllQ4Count
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    mergeSecObj = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    mergeSecObj

    #######################################################################

    event = 'death'
    query = f"""
        SELECT
            t1.t,
            CASE WHEN t2.AllQ1Count <> 0 THEN t1.{event}Q1Count * 1.0 / t2.AllQ1Count ELSE 0 END AS Q1{event}Frac,
            CASE WHEN t2.AllQ2Count <> 0 THEN t1.{event}Q2Count * 1.0 / t2.AllQ2Count ELSE 0 END AS Q2{event}Frac,
            CASE WHEN t2.AllQ3Count <> 0 THEN t1.{event}Q3Count * 1.0 / t2.AllQ3Count ELSE 0 END AS Q3{event}Frac,
            CASE WHEN t2.AllQ4Count <> 0 THEN t1.{event}Q4Count * 1.0 / t2.AllQ4Count ELSE 0 END AS Q4{event}Frac
        FROM
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS {event}Q1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS {event}Q2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS {event}Q3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                dead_after_t
            GROUP BY
                t) AS t1
        INNER JOIN
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS AllQ1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS AllQ2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS AllQ3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS AllQ4Count
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    deadObj = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    deadObj

    #######################################################################

    event = 'birth'

    query = f"""
    SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS {event}Q1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS {event}Q2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS {event}Q3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                born_at_t
            GROUP BY
                t
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    bornObj = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    bornObj

    #######################################################################

    query = f"""
    SELECT
                t,
                SUM(CASE WHEN {category}Q = 1 THEN intensity END) AS Q1Intensity,
                SUM(CASE WHEN {category}Q = 2 THEN intensity END) AS Q2Intensity,
                SUM(CASE WHEN {category}Q = 3 THEN intensity END) AS Q3Intensity,
                SUM(CASE WHEN {category}Q = 4 THEN intensity END) AS Q4Intensity
            FROM
                object_properties
            GROUP BY
                t
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    intDF = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    intDF

    #######################################################################


    query = f"""
    SELECT
                t,
                SUM(intensity)
            FROM
                object_properties
            GROUP BY
                t
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    totInt = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    totInt

    #######################################################################

    query = f"""
    SELECT
                t,
                AVG(CASE WHEN {category}Q = 1 THEN remaining_life END) AS Q1Exp,
                AVG(CASE WHEN {category}Q = 2 THEN remaining_life END) AS Q2Exp,
                AVG(CASE WHEN {category}Q = 3 THEN remaining_life END) AS Q3Exp,
                AVG(CASE WHEN {category}Q = 4 THEN remaining_life END) AS Q4Exp
            FROM
                object_properties
            GROUP BY
                t
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    expDF = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    expDF

    #######################################################################


    query = f"""
    SELECT
                t,
                AVG(CASE WHEN {category}Q = 1 THEN remaining_life END) AS Q1Exp,
                AVG(CASE WHEN {category}Q = 2 THEN remaining_life END) AS Q2Exp,
                AVG(CASE WHEN {category}Q = 3 THEN remaining_life END) AS Q3Exp,
                AVG(CASE WHEN {category}Q = 4 THEN remaining_life END) AS Q4Exp
            FROM
                object_properties
            WHERE
                born_at_t
            GROUP BY
                t
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    expDFnew = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    expDFnew
    #######################################################################


    event = 'density'
    query = f"""
        SELECT
                t,
                AVG(CASE WHEN {category}Q = 1 THEN density END) AS Q1{event},
                AVG(CASE WHEN {category}Q = 2 THEN density END) AS Q2{event},
                AVG(CASE WHEN {category}Q = 3 THEN density END) AS Q3{event},
                AVG(CASE WHEN {category}Q = 4 THEN density END) AS Q4{event}
            FROM
                object_properties
            GROUP BY
                t
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    densityDF = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    densityDF

    #######################################

    #######################################################################


    event = 'intensityInc'

    query = f"""
        SELECT
            t1.t,
            CASE WHEN t2.AllQ1Count <> 0 THEN t1.{event}Q1Count * 1.0 / t2.AllQ1Count ELSE 0 END AS Q1{event}Frac,
            CASE WHEN t2.AllQ2Count <> 0 THEN t1.{event}Q2Count * 1.0 / t2.AllQ2Count ELSE 0 END AS Q2{event}Frac,
            CASE WHEN t2.AllQ3Count <> 0 THEN t1.{event}Q3Count * 1.0 / t2.AllQ3Count ELSE 0 END AS Q3{event}Frac,
            CASE WHEN t2.AllQ4Count <> 0 THEN t1.{event}Q4Count * 1.0 / t2.AllQ4Count ELSE 0 END AS Q4{event}Frac
        FROM
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS {event}Q1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS {event}Q2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS {event}Q3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                intensity_change_next>0 and will_split_next=0 and merge_secondary_at_next_t=0 and merge_primary_at_next_t=0
            GROUP BY
                t) AS t1
        INNER JOIN
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS AllQ1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS AllQ2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS AllQ3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS AllQ4Count
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    intensityIncDF = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    intensityIncDF

    #######################################################################


    event = 'intensityDec'

    query = f"""
        SELECT
            t1.t,
            CASE WHEN t2.AllQ1Count <> 0 THEN t1.{event}Q1Count * 1.0 / t2.AllQ1Count ELSE 0 END AS Q1{event}Frac,
            CASE WHEN t2.AllQ2Count <> 0 THEN t1.{event}Q2Count * 1.0 / t2.AllQ2Count ELSE 0 END AS Q2{event}Frac,
            CASE WHEN t2.AllQ3Count <> 0 THEN t1.{event}Q3Count * 1.0 / t2.AllQ3Count ELSE 0 END AS Q3{event}Frac,
            CASE WHEN t2.AllQ4Count <> 0 THEN t1.{event}Q4Count * 1.0 / t2.AllQ4Count ELSE 0 END AS Q4{event}Frac
        FROM
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS {event}Q1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS {event}Q2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS {event}Q3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS {event}Q4Count
            FROM
                object_properties
            WHERE
                merge_primary_at_next_t=0 AND will_split_next=0 AND merge_secondary_at_next_t=0 AND dead_after_t=0 AND intensity_change_next<0

            GROUP BY
                t) AS t1
        INNER JOIN
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS AllQ1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS AllQ2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS AllQ3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS AllQ4Count
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    intensityDecDF = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    intensityDecDF

    #######################################################################

    event = 'avgIntensity'

    query = f"""
        SELECT
            t1.t,
            CASE WHEN t2.AllQ1Count <> 0 THEN t1.{event}Q1Count * 1.0 / t2.AllQ1Count ELSE 0 END AS Q1AvgInt,
            CASE WHEN t2.AllQ2Count <> 0 THEN t1.{event}Q2Count * 1.0 / t2.AllQ2Count ELSE 0 END AS Q2AvgInt,
            CASE WHEN t2.AllQ3Count <> 0 THEN t1.{event}Q3Count * 1.0 / t2.AllQ3Count ELSE 0 END AS Q3AvgInt,
            CASE WHEN t2.AllQ4Count <> 0 THEN t1.{event}Q4Count * 1.0 / t2.AllQ4Count ELSE 0 END AS Q4AvgInt
        FROM
            (SELECT
                t,
                SUM(CASE WHEN {category}Q = 1 THEN intensity END) AS {event}Q1Count,
                SUM(CASE WHEN {category}Q = 2 THEN intensity END) AS {event}Q2Count,
                SUM(CASE WHEN {category}Q = 3 THEN intensity END) AS {event}Q3Count,
                SUM(CASE WHEN {category}Q = 4 THEN intensity END) AS {event}Q4Count
            FROM
                object_properties
            GROUP BY
                t) AS t1
        INNER JOIN
            (SELECT
                t,
                COUNT(CASE WHEN {category}Q = 1 THEN 1 END) AS AllQ1Count,
                COUNT(CASE WHEN {category}Q = 2 THEN 1 END) AS AllQ2Count,
                COUNT(CASE WHEN {category}Q = 3 THEN 1 END) AS AllQ3Count,
                COUNT(CASE WHEN {category}Q = 4 THEN 1 END) AS AllQ4Count
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    # query = f"""
    # SELECT
    #     t,
    #     AVG(CASE WHEN {category}Q = 1 THEN intensity END) AS Q1AvgInt,
    #     AVG(CASE WHEN {category}Q = 2 THEN intensity END) AS Q2AvgInt,
    #     AVG(CASE WHEN {category}Q = 3 THEN intensity END) AS Q3AvgInt,
    #     AVG(CASE WHEN {category}Q = 4 THEN intensity END) AS Q4AvgInt
    # FROM object_properties
    # GROUP BY t
    # ORDER BY t;
    # """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    avgIntensityDF = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    avgIntensityDF
    # print(avgIntensityDF)
    #######################################################################


    event = 'PeakInt'

    query = f"""
            SELECT
                t,
                AVG(CASE WHEN {category}Q = 1 THEN peak_intensity END) AS {event}Q1,
                AVG(CASE WHEN {category}Q = 2 THEN peak_intensity END) AS {event}Q2,
                AVG(CASE WHEN {category}Q = 3 THEN peak_intensity END) AS {event}Q3,
                AVG(CASE WHEN {category}Q = 4 THEN peak_intensity END) AS {event}Q4
            FROM
                object_properties
            GROUP BY
                t
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    peakIntDF = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    peakIntDF

    #######################################################################
    #######################################################################

    event = 'split'
    query = f"""
                select
                    t1.t,
                    t1.splitcount* 1.0 / t2.totalcount
                as
                    splitFrac
                from
                 (
                    select
                        t,
                        count(*)
                    as
                        splitcount
                    from
                        object_properties
                    where
                        will_split_next and merge_primary_at_next_t=0
                    group by
                        t
                    )
                as
                    t1
                inner join
                 (select t, count(*) as totalcount from object_properties group by t) as t2
                 on t1.t=t2.t
            """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    splitCombObj = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    splitCombObj

    #######################################################################

    event = 'Merge'
    query = f"""
        SELECT
            t1.t,
            t1.{event}Count * 1.0 / t2.AllCount AS {event}Frac
        FROM
            (SELECT
                t,
                COUNT(*) AS {event}Count
            FROM
                object_properties
            WHERE
             merge_primary_at_next_t or merge_secondary_at_next_t
            GROUP BY
                t) AS t1
        INNER JOIN
            (SELECT
                t,
                COUNT(*) AS AllCount
            FROM
                object_properties
            GROUP BY
                t) AS t2
        ON t1.t = t2.t;
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    mergeCombObj = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    mergeCombObj

    #######################################################################

    query = """
                select t1.t, t1.total / t2.countt as average_intensity from

                (select t, sum(intensity) as total from object_properties group by t) as t1
                inner join

                (select t, count(*) as countt from object_properties group by t) as t2
                on t1.t=t2.t
            """
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    avgIntObj = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    avgIntObj

    #######################################################################

    query = f"""
                    SELECT
        t,
            AVG(CASE WHEN {category}Q = 1 THEN size END) AS {category}Q1_sizeAvg,
            AVG(CASE WHEN {category}Q = 2 THEN size END) AS {category}Q4_sizeAvg,
            AVG(CASE WHEN {category}Q = 3 THEN size END) AS {category}Q4_sizeAvg,
            AVG(CASE WHEN {category}Q = 4 THEN size END) AS {category}Q4_sizeAvg
        FROM
            object_properties
        GROUP BY
            t
        ORDER BY
            t;
            """
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    avgSizeObj = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    avgSizeObj



    #######################################################################

    dfs = [splitObj, mergeObj, mergePrimObj, mergeSecObj, deadObj, bornObj,
           totObj, totInt, intDF, peakIntDF, densityDF,
           avgIntensityDF, expDF, expDFnew, intensityIncDF, intensityDecDF, mergeCombObj, avgIntObj, avgSizeObj
           ,splitCombObj
           ]
    output_file = plotsavepath + "/Size_Distribution_of_Events_and_Expectancy_over_Time.csv"
    merged_df = merge_dataframes_and_save(dfs, output_file)

    titles = ['Split Fraction',
              'Total Merge Fraction',
              'Primary merge Fraction',
              'Secondary Merge Fraction',
              'Death Fraction',

              'Birth Count',
              'Total Objects',
              'Total Intensity',
              'Group Intensity',
              'Avg. Peak Intensity',

              'Average Density',
              'Average Intensity',
              'Avg. Life Expectancy',
              'Avg. Life Expectancy of New Objects',
              'Fraction of objects with intensity increase',

              'Fraction of objects with intensity decrease',
               'Merge Fraction, Overall' ,
              'Average Intensity',
              'Average Size',
              'Split Fraction, Overall',
              ]

    y_axs = ['Fraction of Objects of the Same Group',
             'Fraction of Objects of the Same Group',
             'Fraction of Objects of the Same Group',
             'Fraction of Objects of the Same Group',
             'Fraction of Objects of the Same Group',
             'Number of Objects of the Same Group',

             'Number of Objects',
             'Total Intensity',
             'Total Intensity of Each Group',
             'Average Peak Intensity',
             'Average Density',

             'Average Intensity',
             'Life Expectancy in Timepoints',
             'Life Expectancy of New Obj.',
             'Fraction of Objects of the same group',
             'Fraction of Objects of the same group',
             'Fraction of All Objects',
             'Average Intensity',
             'Average Size',
             'Fraction of All Objects'
             ]

    colors = ['blue', 'purple', 'magenta', 'red']

    ix = 4;
    jx = 5
    fig, axes = plt.subplots(ix, jx, figsize=(10 * jx, ix * 10))
    for i in range(ix):
        for j in range(jx):
            # if i > 2 and j > 3:
            #     pass
            # else:
            ax = axes[i, j]
            if i < 1 and j < 4 and j > 0:
                ax.set_ylim(0, 0.3)
            df = dfs[i * jx + j].drop(columns=['t'])
            smooth(df).plot(ax=ax, color=colors)
            #                 ax.axvline(x=acp, color='green', linestyle='--', label='apical constriction point')
            ax.set_title(titles[i * jx + j] + ' (Moving average)', fontsize=15)
            ax.set_xlabel('Timepoint', fontsize=15)
            ax.set_ylabel(y_axs[i * jx + j], fontsize=15)
            ax.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(plotsavepath + '/Size_Distribution_of_Events_and_Expectancy_over_Time.png')
    plt.close()


#######################################################################
# # 555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555
#     dfs = [totObj,  bornObj, deadObj*0.5, splitObj, mergeObj,
#            # splitCombObj, mergeCombObj
#            ]
#     output_file = plotsavepath + "/Size_Distribution_of_Events_and_Expectancy_over_Time.csv"
#     merged_df = merge_dataframes_and_save(dfs, output_file)
#
#     titles = [ '(a) Total Objects',
#               '(b) Birth Count by Size',
#                '(c) Death Fraction by Size',
#                '(d) Split Fraction by Size',
#               '(e) Fusion Fraction by Size',
#                # 'Total Split Fraction',
#                # 'Total Fusion Fraction',
#               ]
#
#     y_axs = ['Number of Objects',
#              'Number of Objects',
#              'Fraction of Objects',
#              'Fraction of Objects',
#              'Fraction of Objects',
#              # 'Fraction of Objects',
#              # 'Fraction of Objects'
#              ]
#
#     colors = ['blue', 'purple', 'magenta', 'red']
#     q_labels = ['S1', 'S2', 'S3', 'S4']
#     n_plots = len(dfs)
#     fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
#     for i in range(n_plots):
#         ax = axes[i]
#         if i  == 1:
#             ax.set_ylim(0, 50)
#         elif i  == 2:
#             ax.set_ylim(0, 0.5)
#         df = dfs[i].drop(columns=['t'])
#         # df.rolling(window=21, min_periods=1).mean().plot(ax=ax, color=colors)
#         for col, color, label in zip(df.columns, colors, q_labels):
#             smoothed = df[col].rolling(window=21, min_periods=1).mean()
#             ax.plot(df.index, smoothed, label=label, color=color)
#         ax.set_title(titles[i] + ' (Moving average)', fontsize=16)
#         ax.set_xlabel('Timepoint', fontsize=16)
#         ax.set_ylabel(y_axs[i], fontsize=16)
#         ax.legend(fontsize=16)
#     plt.tight_layout()
#     plt.savefig(plotsavepath + '/SMOLLLL Size_Distribution_of_Events_and_Expectancy_over_Time.png')
#     plt.close()
# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    output_file = plotsavepath + "/Event_Frequency_Distribution.csv"

    for df in dfs:
        print(df.columns)

    # stop

    fractions1 = pd.concat([splitObj['Q1splitFrac'], mergeObj['Q1MergeFrac'], deadObj['Q1deathFrac'], intensityIncDF['Q1intensityIncFrac'], intensityDecDF['Q1intensityDecFrac']], axis=1)
    fractions1.columns = ['split', 'merge', 'death', 'stayInc', 'stayDec']

    fractions2 = pd.concat(
        [splitObj['Q2splitFrac'], mergeObj['Q2MergeFrac'], deadObj['Q2deathFrac'], intensityIncDF['Q2intensityIncFrac'], intensityDecDF['Q2intensityDecFrac']], axis=1)
    fractions2.columns = ['split', 'merge', 'death', 'stayInc', 'stayDec']

    fractions3 = pd.concat(
        [splitObj['Q3splitFrac'], mergeObj['Q3MergeFrac'], deadObj['Q3deathFrac'], intensityIncDF['Q3intensityIncFrac'], intensityDecDF['Q3intensityDecFrac']], axis=1)
    fractions3.columns = ['split', 'merge', 'death', 'stayInc', 'stayDec']

    fractions4 = pd.concat(
        [splitObj['Q4splitFrac'], mergeObj['Q4MergeFrac'], deadObj['Q4deathFrac'], intensityIncDF['Q4intensityIncFrac'], intensityDecDF['Q4intensityDecFrac']], axis=1)
    fractions4.columns = ['split', 'merge', 'death', 'stayInc', 'stayDec']


    dfs = [fractions1, fractions2, fractions3, fractions4]
    merged_df = merge_dataframes_and_save(dfs, output_file)
    titles = ['Q1', 'Q2', 'Q3', 'Q4']

    y_axs = ['Fraction of Objects',
             'Fraction of Objects',
             'Fraction of Objects',
             'Fraction of Objects',
             'Fraction of Objects',
            ]

    ix = 4; jx=1
    fig, axes = plt.subplots(ix, jx, figsize=(10 * jx, ix * 10))
    for i in range(ix):
        ax = axes[i]
        df = dfs[i]
        smooth(df).plot(kind='line', ax=ax, color=['orange','green','red','blue','magenta'])
        #                 ax.axvline(x=acp, color='green', linestyle='--', label='apical constriction point')
        ax.set_title(titles[i] + ' (Moving Average)', fontsize=15)
        ax.set_xlabel('Timepoint', fontsize=15)
        ax.set_ylabel(y_axs[i], fontsize=15)

    plt.tight_layout()
    plt.savefig(plotsavepath + '/Event_Frequency_Distribution.png')
#######################################################################

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

def getIntensityChange(dbpath, sT, origImgPath, plotsavepath, t1=None, t2=None):
    trackedimagepath = os.path.join(dbpath, 'TrackedCombined.nii')
    samplename = os.path.basename(origImgPath)[:-4]

    query = f"""
    SELECT
                 t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    merge_primary_at_next_t=0 AND will_split_next=0 AND merge_secondary_at_next_t=0 AND dead_after_t=0 AND intensity_change_next>0
                GROUP BY
                    t
                )
                AS
                    t1
                INNER JOIN
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    ((merge_primary_at_t=0) AND (splitted_at_t=0) AND (split_off_at_t=0) AND (born_at_t=0) AND intensity_change_from_prev>0)
                GROUP BY
                    t
                )
                AS
                    t2
                ON
                    t2.t = t1.t+1

    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    IncBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]

    ##################################################################

    query = f"""
    SELECT
                 t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    merge_primary_at_next_t=0 AND will_split_next=0 AND merge_secondary_at_next_t=0 AND dead_after_t=0 AND intensity_change_next<0
                GROUP BY
                    t
                )
                AS
                    t1
                INNER JOIN
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (merge_primary_at_t=0) AND (splitted_at_t=0) AND (split_off_at_t=0) AND (born_at_t=0) AND intensity_change_from_prev<0
                GROUP BY
                    t
                )
                AS
                    t2
                ON
                    t2.t = t1.t+1

    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    DecBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]

    ##################################################################

    query = f"""
                SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    born_at_t
                GROUP BY
                    t
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    BirthBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]

    # ===========================================================================================================

    query = f"""
                SELECT
                    t+1 as t,
                    -SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    dead_after_t
                GROUP BY
                    t
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    DeathBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]
    DeathBy

    ##################################################################


    query = f"""
                SELECT
                   t+1 as t,
                    SUM(intensity) AS Int
                FROM
                    object_properties
                WHERE
                    merge_secondary_at_next_t
                GROUP BY
                    t
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    SecDF = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]
    SecDF

    ##################################################################

    ##################################################################


    query = f"""
                SELECT
                   t+1 as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity_change_next END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity_change_next END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity_change_next END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity_change_next END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    merge_primary_at_next_t and will_split_next=0
                GROUP BY
                    t
    """
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    MergeBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]
    MergeBy

    ##################################################################


    query = f"""
                SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val,
                    SUM(intensity) AS TotalIntensity
                FROM
                    object_properties
                GROUP BY
                    t
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    IntensityDF = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)].drop(columns=['t'])

    ##################################################################


    query = f"""
            SELECT
                t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    will_split_next and merge_primary_at_next_t=0
                GROUP BY
                    t
                )
                AS
                    t1
                INNER JOIN
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (splitted_at_t or split_off_at_t) and merge_primary_at_t=0
                GROUP BY
                    t
                )
                AS
                    t2
                ON
                    t1.t = t2.t-1

    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    SplitBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]

    #     print(SplitBy.loc[20:30,:])

    ##################################################################
    max_t = BirthBy['t'].max()
    # ==================GATHERING===============================================================
    dfs = [SplitBy, MergeBy, BirthBy, DeathBy, IncBy, DecBy]
    for df in dfs:
        df.fillna(0)
        df.set_index('t', inplace=True)
    for df in dfs:
        for i in range(3, max_t + 1):
            if i not in list(df.index):
                df.loc[i, :] = 0
    for df in dfs:
        df.sort_index(inplace=True)
    for df in dfs:
        df.astype(int)
    # ============
    #     SumBy = SplitBy + MergeBy + BirthBy + DeathBy + IncBy + DecBy
    # ===========================================================================================================

    dfs = [SplitBy, MergeBy, BirthBy, DeathBy, IncBy, DecBy]
    alldf = pd.concat(
        [SplitBy.sum(axis=1), MergeBy.sum(axis=1), BirthBy.sum(axis=1), DeathBy.sum(axis=1), IncBy.sum(axis=1),
         DecBy.sum(axis=1)], axis=1)
    alldf = alldf.sum(axis=1).to_frame()

    output_file = plotsavepath + '/' + 'BY_' + category.upper() + '_change_in_Intensity.csv'
    merged_df = merge_dataframes_and_save(dfs, output_file)

    cumsum = alldf.cumsum()
    cumsum.columns = ['cumsum']

    alldf.columns = ['Intensity']
    alldf['Intensity'] = alldf['Intensity'] - SecDF['Int']
    alldf = smooth(alldf)

    titles = [
        'Intensity change of Split objects',
        'Intensity change of Merge objects',
        'Intensity of Born objects',
        'Intensity when dead',
        'Intensity change of Inactive, Increase',
        'Intensity change of Inactive, Decrease',

    ]

    y_axs = ['Intensity'] * len(titles)

    colors = ['blue', 'purple', 'magenta', 'red']

    ix = 2;
    jx = 3;
    countdf = 0
    fig, axes = plt.subplots(ix, jx, figsize=(15 * jx, ix * 10))
    for i in range(ix):
        for j in range(jx):
            if i * jx + j < len(dfs):
                countdf += 1
                ax = axes[i, j]
                df = dfs[i * jx + j]  # .drop(columns=['t'])
                smooth(df).plot(ax=ax, color=colors)
                ax.set_title(titles[i * jx + j] + ' (Moving Average)', fontsize=15)
                ax.set_xlabel('timepoint', fontsize=15)
                ax.set_ylabel(y_axs[i * jx + j], fontsize=15)
                ax.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(plotsavepath + '/' + 'BY_' + category.upper() + '_change_in_Intensity.png')
# ====================================================================================================
    # Fraction is just a variable name, doesn't mean anything.
    Q = 'CondQ1Val'
    fractions1 = pd.concat([SplitBy[Q], MergeBy[Q], BirthBy[Q], DeathBy[Q], IncBy[Q], DecBy[Q]], axis=1)
    fractions1.columns = ['split', 'merge', 'birth', 'death', 'stayInc', 'stayDec']
    Q = 'CondQ2Val'
    fractions2 = pd.concat([SplitBy[Q], MergeBy[Q], BirthBy[Q], DeathBy[Q], IncBy[Q], DecBy[Q]], axis=1)
    fractions2.columns = ['split', 'merge', 'birth', 'death', 'stayInc', 'stayDec']
    Q = 'CondQ3Val'
    fractions3 = pd.concat([SplitBy[Q], MergeBy[Q], BirthBy[Q], DeathBy[Q], IncBy[Q], DecBy[Q]], axis=1)
    fractions3.columns = ['split', 'merge', 'birth', 'death', 'stayInc', 'stayDec']
    Q = 'CondQ4Val'
    fractions4 = pd.concat([SplitBy[Q], MergeBy[Q], BirthBy[Q], DeathBy[Q], IncBy[Q], DecBy[Q]], axis=1)
    fractions4.columns = ['split', 'merge', 'birth', 'death', 'stayInc', 'stayDec']

    f1 = smooth(fractions1, trim=True)
    f2 = smooth(fractions2, trim=True)
    f3 = smooth(fractions3, trim=True)
    f4 = smooth(fractions4, trim=True)

    dfs = [f1, f2, f3, f4, alldf, IntensityDF]

    ##########################################added####
    f1sum = f1.sum(axis=1)
    f2sum = f2.sum(axis=1)
    f3sum = f3.sum(axis=1)
    f4sum = f4.sum(axis=1)
    sumdfs = [f1sum, f2sum, f3sum, f4sum]
    ix = 4;
    jx = 1;
    countdf = 0
    sumtitle = ['Q1', 'Q2', 'Q3', 'Q4']

    print(f1sum, f2sum,f3sum, f4sum) #%%%%%%%%%%%%%%%%%%%%%%%%%
    fig, axes = plt.subplots(ix, jx, figsize=(20, 30))
    for i in range(ix):
        ax = axes[i]
        df = sumdfs[i]

        df.plot(ax=ax, kind='bar', color=colors[i])
        ax.axhline(y=0, c="black", linewidth=0.5, zorder=0)
        ax.set_ylabel('Total change')
        ax.set_title(sumtitle[i])
        ax.set_xlabel('Time (t)')

        ax.legend(title='Total change', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    plt.tight_layout()
    plt.savefig(
        plotsavepath + '/' + 'BY_' + category.upper() + '_change_in_Intensity__SUM__BAR.png')
    ##########################################added####

    title = ['Q1', 'Q2', 'Q3', 'Q4', 'Overall Change', 'Total Intensity']
    kinds = ['bar', 'bar', 'bar', 'bar', 'bar', 'line']

    ix = 6;
    jx = 1;
    countdf = 0
    fig, axes = plt.subplots(ix, jx, figsize=(20, 30))
    for i in range(ix):
        ax = axes[i]
        df = dfs[i]

        if i == 5:
            smooth(df).plot(ax=ax, color=['blue', 'purple', 'magenta', 'red', 'black'])
        else:
            df.plot(ax=ax, kind=kinds[i], stacked=True if i < 4 else False, color=color6 if i < 4 else 'black')
            ax.set_ylabel('Change Value')
        ax.set_title(title[i])
        ax.set_xlabel('Time (t)')

        ax.legend(title='Change', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    plt.tight_layout()
    plt.savefig(plotsavepath + '/' + 'BY_' + category.upper() + '_change_in_Intensity_BAR.png')

    ix = 4;
    jx = 1;
    countdf = 0
    fig, axes = plt.subplots(ix, jx, figsize=(20, 30))
    for i in range(ix):
        ax = axes[i]
        df = dfs[i]

        df.plot(ax=ax, kind='line', color=color6 if i < 4 else 'black')
        ax.axhline(y=0, c="black", linewidth=0.5, zorder=0)
        ax.set_ylabel('Change value')
        ax.set_title(title[i])
        ax.set_xlabel('Time (t)')

        ax.legend(title='Change', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    plt.tight_layout()
    plt.savefig(plotsavepath + '/' + 'BY_' + category.upper() + '_change_in_Intensity_LINE.png')


####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################


def getIntensityChangeAvg(dbpath, sT, origImgPath, plotsavepath, t1=None, t2=None):
    trackedimagepath = os.path.join(dbpath, 'TrackedCombined.nii')
    samplename = os.path.basename(origImgPath)[:-4]

    query = f"""
    SELECT
                 t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    AVG(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    merge_primary_at_next_t=0 AND will_split_next=0 AND merge_secondary_at_next_t=0 AND dead_after_t=0 AND intensity_change_next>0
                GROUP BY
                    t
                )
                AS
                    t1
                INNER JOIN
                (SELECT
                    t as t,
                    AVG(COALESCE(CASE WHEN previous_{category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    AVG(COALESCE(CASE WHEN previous_{category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    AVG(COALESCE(CASE WHEN previous_{category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    AVG(COALESCE(CASE WHEN previous_{category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (merge_primary_at_t=0) AND (splitted_at_t=0) AND (split_off_at_t=0) AND (born_at_t=0) AND intensity_change_from_prev>0
                GROUP BY
                    t
                )
                AS
                    t2
                ON
                    t2.t = t1.t+1

    """
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    IncBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]

    ##################################################################


    query = f"""
    SELECT
                 t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    AVG(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    merge_primary_at_next_t=0 AND will_split_next=0 AND merge_secondary_at_next_t=0 AND dead_after_t=0 AND intensity_change_next<0
                GROUP BY
                    t
                )
                AS
                    t1
                INNER JOIN
                (SELECT
                    t as t,
                    AVG(COALESCE(CASE WHEN previous_{category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    AVG(COALESCE(CASE WHEN previous_{category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    AVG(COALESCE(CASE WHEN previous_{category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    AVG(COALESCE(CASE WHEN previous_{category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (merge_primary_at_t=0) AND (splitted_at_t=0) AND (split_off_at_t=0) AND (born_at_t=0) AND intensity_change_from_prev<0
                GROUP BY
                    t
                )
                AS
                    t2
                ON
                    t2.t = t1.t+1

    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    DecBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]

    ##################################################################

    query = f"""
                SELECT
                    t as t,
                    AVG(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    born_at_t
                GROUP BY
                    t
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    BirthBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]

    # ===========================================================================================================

    query = f"""
                SELECT
                    t+1 as t,
                    -AVG(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    -AVG(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    -AVG(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    -AVG(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    dead_after_t
                GROUP BY
                    t
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    DeathBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]
    DeathBy

    ##################################################################


    query = f"""
                SELECT
                   t+1 as t,
                    AVG(COALESCE(CASE WHEN {category}Q = 1 THEN intensity_change_next END, 0)) AS CondQ1Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 2 THEN intensity_change_next END, 0)) AS CondQ2Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 3 THEN intensity_change_next END, 0)) AS CondQ3Val,
                    AVG(COALESCE(CASE WHEN {category}Q = 4 THEN intensity_change_next END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    merge_primary_at_next_t and will_split_next=0
                GROUP BY
                    t
    """
    # and will_split_next=0

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    MergeBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]
    MergeBy

    ##################################################################


    query = f"""
                SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val,
                    SUM(intensity) AS TotalIntensity
                FROM
                    object_properties
                GROUP BY
                    t
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    IntensityDF = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)].drop(columns=['t'])

    ##################################################################

    ##################################################################

    # using 2x the average for split, as the splitted and split_off are counted individually and will be divided
    # by twice the number of events -- yeeeeeaaaaah
    query = f"""
            SELECT
            tx.t as t,
            COALESCE(CASE WHEN t3.q1c>0 THEN tx.CondQ1Val/t3.q1c END, 0) as CondQ1Val,
            COALESCE(CASE WHEN t3.q2c>0 THEN tx.CondQ2Val/t3.q2c END, 0) as CondQ2Val,
            COALESCE(CASE WHEN t3.q3c>0 THEN tx.CondQ3Val/t3.q3c END, 0) as CondQ3Val,
            COALESCE(CASE WHEN t3.q4c>0 THEN tx.CondQ4Val/t3.q4c END, 0) as CondQ4Val

            FROM

            (SELECT
                t2.t as t,
                (-t1.CondQ1Val + t2.CondQ1Val) as CondQ1Val,
                (-t1.CondQ2Val + t2.CondQ2Val) as CondQ2Val,
                (-t1.CondQ3Val + t2.CondQ3Val) as CondQ3Val,
                (-t1.CondQ4Val + t2.CondQ4Val) as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    will_split_next and merge_primary_at_next_t=0
                GROUP BY
                    t
                )
                AS
                    t1
                INNER JOIN
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN previous_{category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (splitted_at_t or split_off_at_t) and merge_primary_at_t=0
                GROUP BY
                    t
                )
                AS
                    t2
                ON
                    t1.t = t2.t-1
                )
                AS
                    tx
                INNER JOIN
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN 1 END, 0)) AS q1c,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN 1 END, 0)) AS q2c,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN 1 END, 0)) AS q3c,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN 1 END, 0)) AS q4c
                FROM
                    object_properties
                WHERE
                    will_split_next and merge_primary_at_next_t=0
                GROUP BY
                    t
                )
                AS
                    t3
                ON
                    t3.t = tx.t-1

    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    SplitBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]

    smooth(SplitBy['CondQ4Val']).plot()
    #     stop

    #     print(SplitBy.loc[20:30,:])
    #     print(IncBy.loc[20:30,:])

    ##################################################################
    max_t = BirthBy['t'].max()
    # ==================GATHERING===============================================================
    dfs = [SplitBy, MergeBy, BirthBy, DeathBy, IncBy, DecBy]
    for df in dfs:
        df.fillna(0)
        df.set_index('t', inplace=True)
    for df in dfs:
        for i in range(3, max_t + 1):
            if i not in list(df.index):
                df.loc[i, :] = 0
    for df in dfs:
        df.sort_index(inplace=True)
    for df in dfs:
        df.astype(int)
    # ============
    #     SumBy = SplitBy + MergeBy + BirthBy + DeathBy + IncBy + DecBy
    # ===========================================================================================================

    dfs = [SplitBy, MergeBy, BirthBy, DeathBy, IncBy, DecBy]

    output_file = plotsavepath + '/' + 'BY_' + category.upper() + '_PER_EVENT_change_in_Intensity.csv'
    merged_df = merge_dataframes_and_save(dfs, output_file)

    titles = [
        'Avg. intensity change of Split objects',
        'Avg. intensity change of Merge objects',
        'Avg. intensity of Born objects',
        'Avg. intensity when dead',
        'Avg. intensity change of Inactive, Increase',
        'Avg. intensity change of Inactive, Decrease']

    y_axs = ['Intensity'] * len(titles)

    colors = ['blue', 'purple', 'magenta', 'red']

    ix = 2;
    jx = 3;
    countdf = 0
    fig, axes = plt.subplots(ix, jx, figsize=(15 * jx, ix * 10))
    for i in range(ix):
        for j in range(jx):
            if i * jx + j < len(dfs):
                countdf += 1
                ax = axes[i, j]
                df = dfs[i * jx + j]  # .drop(columns=['t'])
                smooth(df).plot(ax=ax, color=colors)
                #                 ax.axvline(x=acp, color='green', linestyle='--', label='apical constriction point')
                ax.set_title(titles[i * jx + j] + ' (Moving Average)', fontsize=15)
                ax.set_xlabel('timepoint', fontsize=15)
                ax.set_ylabel(y_axs[i * jx + j], fontsize=15)
                ax.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(plotsavepath + '/' + 'BY_' + category.upper() + '_PER_EVENT_change_in_Intensity.png')
    #     plt.show()

    ##################################################################

    # Fraction is just a variable copied from somewhere, doesn't mean anything.
    Q = 'CondQ1Val'
    fractions1 = pd.concat([SplitBy[Q], MergeBy[Q], BirthBy[Q], DeathBy[Q], IncBy[Q], DecBy[Q]], axis=1)
    fractions1.columns = ['split', 'merge', 'birth', 'death', 'stayInc', 'stayDec']
    Q = 'CondQ2Val'
    fractions2 = pd.concat([SplitBy[Q], MergeBy[Q], BirthBy[Q], DeathBy[Q], IncBy[Q], DecBy[Q]], axis=1)
    fractions2.columns = ['split', 'merge', 'birth', 'death', 'stayInc', 'stayDec']
    Q = 'CondQ3Val'
    fractions3 = pd.concat([SplitBy[Q], MergeBy[Q], BirthBy[Q], DeathBy[Q], IncBy[Q], DecBy[Q]], axis=1)
    fractions3.columns = ['split', 'merge', 'birth', 'death', 'stayInc', 'stayDec']
    Q = 'CondQ4Val'
    fractions4 = pd.concat([SplitBy[Q], MergeBy[Q], BirthBy[Q], DeathBy[Q], IncBy[Q], DecBy[Q]], axis=1)
    fractions4.columns = ['split', 'merge', 'birth', 'death', 'stayInc', 'stayDec']

    f1 = smooth(fractions1, trim=True)
    f2 = smooth(fractions2, trim=True)
    f3 = smooth(fractions3, trim=True)
    f4 = smooth(fractions4, trim=True)

    dfs = [f1, f2, f3, f4]
    title = ['Q1', 'Q2', 'Q3', 'Q4']

    ix = 4;
    jx = 1;
    countdf = 0
    fig, axes = plt.subplots(ix, jx, figsize=(20, 20))
    for i in range(ix):
        ax = axes[i]
        df = dfs[i]
        df.plot(ax=ax, kind='bar', stacked=True, color=color6)
        ax.set_title(title[i])
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Change value')
        ax.legend(title='Change', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    plt.tight_layout()
    plt.savefig(
        plotsavepath + '/' + 'BY_' + category.upper() + '_PER_EVENT_change_in_Intensity_BAR.png')

    ix = 4;
    jx = 1;
    countdf = 0
    fig, axes = plt.subplots(ix, jx, figsize=(20, 30))
    for i in range(ix):
        ax = axes[i]
        df = dfs[i]

        df.plot(ax=ax, kind='line', color=color6 if i < 4 else 'black')
        ax.axhline(y=0, c="black", linewidth=0.5, zorder=0)
        ax.set_ylabel('Change value')
        ax.set_title(title[i])
        ax.set_xlabel('Time (t)')

        ax.legend(title='Change', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    plt.tight_layout()
    plt.savefig(
        plotsavepath + '/' + 'BY_' + category.upper() + '_PER_EVENT_change_in_Intensity_LINE.png')


####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

def getIntensityChangeContribution(dbpath, sT, origImgPath, plotsavepath, t1=None, t2=None):
    trackedimagepath = os.path.join(dbpath, 'TrackedCombined.nii')

    samplename = os.path.basename(origImgPath)[:-4]

    colors = ['blue', 'purple', 'magenta', 'red']
    color5 = ['green'] + colors

    query = f"""
             SELECT
                    t as t,
                    SUM(CASE WHEN {category}Q = 1 THEN intensity END) AS CondQ1Val,
                    SUM(CASE WHEN {category}Q = 2 THEN intensity END) AS CondQ2Val,
                    SUM(CASE WHEN {category}Q = 3 THEN intensity END) AS CondQ3Val,
                    SUM(CASE WHEN {category}Q = 4 THEN intensity END) AS CondQ4Val
                FROM
                    object_properties
                GROUP BY
                    t

    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    TotalIntensity = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    TotalIntensity

    # ===========================================================================================================


    query = f"""
            SELECT t4.t as t,
                    t4.CondQ1Val - t5.CondQ1Val as CondQ1Val,
                    t4.CondQ2Val - t5.CondQ2Val as CondQ2Val,
                    t4.CondQ3Val - t5.CondQ3Val as CondQ3Val,
                    t4.CondQ4Val - t5.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                GROUP BY
                    t
                )
                AS
                    t4
                INNER JOIN
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                GROUP BY
                    t
                )
                AS
                    t5
                ON
                    t4.t=t5.t+1

    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    RealBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    max_t = RealBy['t'].max()
    # RealBy=RealBy.drop(columns=['t'])
    RealBy, max_t

    # ===========================================================================================================


    query = f"""
            SELECT
                t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    will_split_next and merge_primary_at_next_t=0
                GROUP BY
                    t
                )
                AS
                    t1
                INNER JOIN
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (splitted_at_t or split_off_at_t) and merge_primary_at_t=0
                GROUP BY
                    t
                )
                AS
                    t2
                ON
                    t1.t = t2.t-1

    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    color5 = ['green', 'blue', 'purple', 'magenta', 'red']
    colors = ['blue', 'purple', 'magenta', 'red']
    SplitBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    SplitBy

    # ===========================================================================================================

    query = f"""
            SELECT
                t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                   t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (merge_primary_at_next_t or merge_secondary_at_next_t) and will_split_next=0
                GROUP BY
                    t
                )
                AS
                    t1
                LEFT JOIN
                (SELECT
                   t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (splitted_at_t=0 and split_off_at_t=0) and merge_primary_at_t
                GROUP BY
                    t
                )
                AS
                    t2
                ON
                    t1.t = t2.t-1
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    color5 = ['green', 'blue', 'purple', 'magenta', 'red']
    colors = ['blue', 'purple', 'magenta', 'red']
    MergeBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    MergeBy

    # ===========================================================================================================


    query = f"""
            SELECT
                t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    merge_primary_at_next_t and will_split_next
                GROUP BY
                    t
                )
                AS
                    t1
                INNER JOIN
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (splitted_at_t or split_off_at_t) and merge_primary_at_t
                GROUP BY
                    t
                )
                AS
                    t2
                ON
                    t1.t = t2.t-1

    """

    query = f"""
            -- Create a temporary table with a sequence of integers
            WITH RECURSIVE sequence AS (
                SELECT MIN(t) AS t
                FROM (
                    SELECT MIN(t) AS t FROM object_properties
                    UNION ALL
                    SELECT MIN(t) - 1 AS t FROM object_properties
                ) AS initial_values
                UNION ALL
                SELECT t + 1
                FROM sequence
                WHERE t < (
                    SELECT MAX(t)
                    FROM (
                        SELECT MAX(t) AS t FROM object_properties
                        UNION ALL
                        SELECT MAX(t) + 1 AS t FROM object_properties
                    ) AS max_values
                )
            ),

            t1 AS (
                SELECT
                    t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    merge_primary_at_next_t AND will_split_next
                GROUP BY
                    t
            ),

            t2 AS (
                SELECT
                    t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (splitted_at_t OR split_off_at_t) AND merge_primary_at_t
                GROUP BY
                    t
            )

            SELECT
                sequence.t,
                COALESCE(t2.CondQ1Val, 0) - COALESCE(t1.CondQ1Val, 0) AS CondQ1Val,
                COALESCE(t2.CondQ2Val, 0) - COALESCE(t1.CondQ2Val, 0) AS CondQ2Val,
                COALESCE(t2.CondQ3Val, 0) - COALESCE(t1.CondQ3Val, 0) AS CondQ3Val,
                COALESCE(t2.CondQ4Val, 0) - COALESCE(t1.CondQ4Val, 0) AS CondQ4Val
            FROM
                sequence
            LEFT JOIN
                t1 ON sequence.t = t1.t
            LEFT JOIN
                t2 ON sequence.t = t2.t - 1
            ORDER BY
                sequence.t;

    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    DoubleBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    DoubleBy

    # ===========================================================================================================

    query = f"""
                SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    born_at_t
                GROUP BY
                    t
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    BirthBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    BirthBy

    # ===========================================================================================================


    query = f"""
                SELECT
                    t+1 as t,
                    -SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    dead_after_t
                GROUP BY
                    t
    """
    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    DeathBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    DeathBy

    # ===========================================================================================================


    query = f"""
                SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    (previous_{category}Q != {category}Q) AND (merge_primary_at_t=0) AND (splitted_at_t=0) AND (split_off_at_t=0) AND (born_at_t=0)
                GROUP BY
                    t

    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    MigrateInBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    MigrateInBy

    # ===========================================================================================================



    query = f"""
                SELECT
                    t+1 as t,
                    -SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    -SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    next_{category}Q != {category}Q AND merge_primary_at_next_t=0 AND will_split_next=0 AND merge_secondary_at_next_t=0 AND dead_after_t=0
                GROUP BY
                    t
    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    MigrateOutBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    MigrateOutBy

    # ===========================================================================================================

    query = f"""
    SELECT
                t2.t as t,
                -t1.CondQ1Val + t2.CondQ1Val as CondQ1Val,
                -t1.CondQ2Val + t2.CondQ2Val as CondQ2Val,
                -t1.CondQ3Val + t2.CondQ3Val as CondQ3Val,
                -t1.CondQ4Val + t2.CondQ4Val as CondQ4Val
            FROM
                (SELECT
                    t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    next_{category}Q = {category}Q AND merge_primary_at_next_t=0 AND will_split_next=0 AND merge_secondary_at_next_t=0 AND dead_after_t=0
                GROUP BY
                    t
                )
                AS
                    t1
                INNER JOIN
                (SELECT
                   t as t,
                    SUM(COALESCE(CASE WHEN {category}Q = 1 THEN intensity END, 0)) AS CondQ1Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 2 THEN intensity END, 0)) AS CondQ2Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 3 THEN intensity END, 0)) AS CondQ3Val,
                    SUM(COALESCE(CASE WHEN {category}Q = 4 THEN intensity END, 0)) AS CondQ4Val
                FROM
                    object_properties
                WHERE
                    previous_{category}Q = {category}Q AND merge_primary_at_t=0 AND splitted_at_t=0 AND split_off_at_t=0 AND born_at_t=0
                GROUP BY
                    t
                )
                AS
                    t2
                ON
                    t1.t = t2.t-1

    """

    conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
    columns, results = dynq_execute(conn, t1, t2, query)
    conn.close()

    newDF = pd.DataFrame(results, columns=columns)

    StayBy = newDF[(newDF['t'] > 2) & (newDF['t'] >= t1) & (newDF['t']<=t2)]  # .drop(columns=['t'])
    StayBy

    # ==================GATHERING===============================================================
    dfs = [SplitBy, MergeBy, DoubleBy, StayBy, MigrateInBy, MigrateOutBy, BirthBy, DeathBy]
    for df in dfs:
        df.fillna(0)
        df.set_index('t', inplace=True)
    for df in dfs:
        for i in range(3, max_t + 1):
            if i not in list(df.index):
                df.loc[i, :] = 0
    for df in dfs:
        df.sort_index(inplace=True)
    for df in dfs:
        df.astype(int)
    # ============

    SumBy = SplitBy + MergeBy + DoubleBy + StayBy + MigrateInBy + MigrateOutBy + BirthBy + DeathBy
    RealBy.set_index('t', inplace=True)

    # ===========================================================================================================

    dfs = [TotalIntensity, SplitBy, MergeBy, DoubleBy, BirthBy, DeathBy, MigrateInBy, MigrateOutBy, StayBy, SumBy,
           RealBy]

    output_file = plotsavepath + '/' + '(CARE WHERE OBJS END UP)BY_' + category.upper() + '_CONTRIBUTION_TO_GROUP_INTENSITY_CHANGE.csv'
    merged_df = merge_dataframes_and_save(dfs, output_file)

    titles = ['Total Intensity',
              'Intensity change due to Split',
              'Intensity change due to Merge',
              'Intensity change due to Split + Merge',
              'Intensity change due to Birth',
              'Intensity change due to Death',
              'Intensity change due to Migrate In',
              'Intensity change due to Migrate Out',
              'Intensity change due to Stay',
              'SUM of Changes',
              'Real Change']

    y_axs = ['Intensity'] * len(titles)

    colors = ['blue', 'purple', 'magenta', 'red']

    ix = 3;
    jx = 4;
    countdf = 0
    fig, axes = plt.subplots(ix, jx, figsize=(15 * jx, ix * 10))
    for i in range(ix):
        for j in range(jx):
            if i * jx + j < len(dfs):
                countdf += 1
                ax = axes[i, j]
                df = dfs[i * jx + j]  # .drop(columns=['t'])
                smooth(df).plot(ax=ax, color=colors)
                #                 ax.axvline(x=acp, color='green', linestyle='--', label='apical constriction point')
                ax.set_title(titles[i * jx + j] + ' (Moving average)', fontsize=15)
                ax.set_xlabel('Timepoint', fontsize=15)

                ax.set_ylabel(y_axs[i * jx + j], fontsize=15)
                ax.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(
        plotsavepath + '/' + '(CARE WHERE OBJS END UP)BY_' + category.upper() + '_CONTRIBUTION_TO_GROUP_INTENSITY_CHANGE.png')

def niftireadu32(arg):
    return np.asarray(nib.load(arg).dataobj).astype(np.uint32).squeeze()

from matplotlib.patches import PathPatch
from matplotlib.path import Path


def getHistogramUno(dbpath, plotsavepath, t1=None, t2=None):
    dbfile = os.path.join(dbpath, "ObjectsProperties.db")

    # ---- read sizes for quantiles + p99 cap ----
    conn = sqlite3.connect(dbfile)
    query = f"SELECT size FROM object_properties WHERE t >= {t1} AND t <= {t2}"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print(f"[getHistogramUno] No rows for t in [{t1}, {t2}]")
        return

    p99 = df["size"].quantile(0.999)
    p99 = float(np.ceil(p99) + 10)

    q_1 = int(df["size"].quantile(0.25))
    q_2 = int(df["size"].quantile(0.50))
    q_3 = int(df["size"].quantile(0.75))
    print(f"QUANTILES T range:  {q_1},{q_2},{q_3}")

    # ---- read (t, size) ----
    conn = sqlite3.connect(dbfile)
    query = f"SELECT t, size FROM object_properties WHERE t >= {t1} AND t <= {t2}"
    newDF = pd.read_sql_query(query, conn)
    conn.close()

    if newDF.empty:
        print(f"[getHistogramUno] No (t,size) rows for t in [{t1}, {t2}]")
        return

    # Group sizes by actual t
    dict1 = newDF.groupby("t")["size"].apply(list).to_dict()
    t_list = sorted(dict1.keys())             # <-- FIX: real timepoints, sorted
    data = [dict1[t] for t in t_list]

    # ---- consistent color normalization across all timepoints ----
    global_max_count = 1
    for vals in data:
        if len(vals) == 0:
            continue
        _, counts = np.unique(vals, return_counts=True)
        if len(counts) > 0:
            global_max_count = max(global_max_count, int(counts.max()))

    norm = plt.Normalize(vmin=1, vmax=global_max_count)

    # ---- plot ----
    plt.figure(figsize=(18, 9))

    # keep your rainbow-ish intent; but don't overwrite it afterward
    n_bins = max(2, len(t_list))
    colors = ["Violet", "Blue", "Cyan", "Green", "Yellow", "Orange", "Red"]
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors, N=n_bins)

    # custom wide marker
    custom_marker = Path(
        [(-1, -0.6), (1, -0.6), (1, 0.6), (-1, 0.6), (-1, -0.6)],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
    )

    scatter = None
    for t, values in zip(t_list, data):
        if len(values) == 0:
            continue
        unique, counts = np.unique(values, return_counts=True)
        count_map = dict(zip(unique, counts))                 # O(N) lookup
        cvals = np.array([count_map[v] for v in values])

        scatter = plt.scatter(
            np.full(len(values), t),                          # <-- FIX: x is actual t
            values,
            c=cvals,
            cmap=cmap,
            norm=norm,
            marker=custom_marker,
            s=10,
            edgecolor="none",
            alpha=0.7,
        )

    # Save the "wide" CSV like before (columns are actual t's, not assumed range)
    keys_ = t_list
    values_ = zip_longest(*[dict1[t] for t in t_list], fillvalue=None)
    df_ = pd.DataFrame(values_, columns=keys_)
    os.makedirs(plotsavepath, exist_ok=True)
    df_.to_csv(os.path.join(plotsavepath, "sizeHistogramNotBucket.csv"), index=False)

    # Colorbar
    if scatter is not None:
        cbar = plt.colorbar(scatter)
        cbar.set_label("Count", fontsize=16)
        cbar.ax.tick_params(labelsize=12)

    # X ticks: label every 10th timepoint (by index), but show the real t value
    tick_idx = [i for i in range(len(t_list)) if (i + 1) % 10 == 0]
    tick_x = [t_list[i] for i in tick_idx]
    plt.xticks(tick_x, [str(x) for x in tick_x], fontsize=12)

    # Quartile lines
    plt.axhline(y=q_1, color="b", linestyle="--", lw=0.5, label="Q1")
    plt.axhline(y=q_2, color="g", linestyle="--", lw=0.5, label="Q2")
    plt.axhline(y=q_3, color="r", linestyle="--", lw=0.5, label="Q3")

    # Quartile textbox
    text_str = f"Q1 = {q_1}\nQ2 = {q_2}\nQ3 = {q_3}"
    plt.text(
        0.98, 0.98, text_str,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Y-lims and labels
    plt.ylim(0, 1000)
    plt.xlabel("Timepoint (t from DB)", fontsize=16)
    plt.ylabel("Object size (pixels)", fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.savefig(os.path.join(plotsavepath, "sizeHistogramNotBucket.png"), dpi=200, bbox_inches="tight")

    plt.ylim(0, p99)
    plt.savefig(os.path.join(plotsavepath, "sizeHistogram_p99.png"), dpi=200, bbox_inches="tight")
    plt.close()

# def getHistogramUno(dbpath, plotsavepath, t1=None, t2=None):
#
#     conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
#     query = f"""
#         SELECT size FROM object_properties WHERE t >= {t1} AND t <= {t2}
#         """
#     df = pd.read_sql_query(query, conn)
#     conn.close()
#
#     p99 = df['size'].quantile(0.999)
#     p99 = np.ceil(p99) + 10
#
#     q_1 = int(df['size'].quantile(0.25))
#     q_2 = int(df['size'].quantile(0.50))
#     q_3 = int(df['size'].quantile(0.75))
#
#     print(f"QUANTILES T range:  {q_1},{q_2},{q_3}")
#
#     conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
#     cursor = conn.cursor()
#     query = f"""
#         SELECT MAX(size) AS max_size FROM object_properties WHERE t >= {t1} AND t <= {t2}
#         """
#     cursor.execute(query)
#     result = cursor.fetchone()
#     conn.close()
#     max_size = result[0] if result else None
#
#     # max_size = min(max_size, 1000)
#     # with open(os.path.join(dbpath, 'quartiles.txt'), 'r') as f:
#     #     line = f.read().strip()
#     #     q_1, q_2, q_3 = map(int, line.split(','))
#     # print(f"QUANTILES SAVED:  {q_1},{q_2},{q_3}")
#
#     conn = sqlite3.connect(os.path.join(dbpath, 'ObjectsProperties.db'))
#     cursor = conn.cursor()
#     query = f"""
#         select t,size from object_properties  where t>={t1} and t<={t2}
#         """
#     cursor.execute(query)
#     columns = [column[0] for column in cursor.description]
#     results = cursor.fetchall()
#     newDF = pd.DataFrame(results)
#     newDF.columns = columns
#     conn.close()
#     dict1 = newDF.groupby('t')['size'].apply(list).to_dict()
#     data = []
#     for akey in dict1.keys():
#         data.append(dict1[akey])
#
#     plt.figure(figsize=(18, 9))
#     cmap = plt.get_cmap('rainbow')
#     n_bins = t2 - t1 + 1
#     cmap_name = 'rainbow'
#     colors = ["Violet", "Blue", "Cyan", "Green", "Yellow", "Orange", "Red"]
#     cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
#     cmap = plt.get_cmap('rainbow')
#
#     # Define a custom wide marker
#     custom_marker = Path(
#         [
#             # (-1, -0.3), (1, -0.3), (1, 0.3), (-1, 0.3), (-1, -0.3)
#         (-1, -0.6), (1, -0.6), (1, 0.6), (-1, 0.6), (-1, -0.6)
#         ],
#         [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
#     )
#
#     for time_index, values in enumerate(data):
#         # Count occurrences of each value
#         unique, counts = np.unique(values, return_counts=True)
#         scatter = plt.scatter([time_index] * len(values), values,
#                               c=[counts[np.where(unique == v)[0][0]] for v in values],
#                               cmap=cmap, marker=custom_marker, s=10, edgecolor='none', alpha=0.7,
#                               label=f'Timepoint {time_index + 1}')
#
#     keys_ = dict1.keys()
#     values_ = zip_longest(*dict1.values(), fillvalue=None)
#     df_ = pd.DataFrame(values_, columns=keys_)
#     df_.to_csv(plotsavepath + '/' + 'sizeHistogramNotBucket.csv')
#     # newDF.to_csv(plotsavepath + '/' + 'sizeHistogramNotBucket.csv')
#
#     # Add color bar to show the count scale
#     cbar = plt.colorbar(scatter, label='count')
#     cbar.set_label('Count', fontsize=16)
#     cbar.ax.tick_params(labelsize=12)
#     plt.xticks(
#         [i for i in range(len(data)) if (i + 1) % 10 == 0],
#         [f'{t1 + i }' for i in range(len(data)) if (i + 1) % 10 == 0],
#         fontsize=12
#     )
#
#     plt.axhline(y=q_1, color='b', linestyle='--', lw='0.5', label='Q1')
#     plt.axhline(y=q_2, color='g', linestyle='--', lw='0.5', label='Q2')
#     plt.axhline(y=q_3, color='r', linestyle='--', lw='0.5', label='Q3')
#
#     # Add text box in top right corner with quartile values
#     text_str = f'Q1 = {q_1}\nQ2 = {q_2}\nQ3 = {q_3}'
#     plt.text(0.98, 0.98, text_str, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
#              horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#
#     # plt.ylim(0, max_size+10)
#     plt.ylim(0, 1000)
#     plt.xlabel('Timepoint',fontsize=16)
#     plt.ylabel('Object size', fontsize=16)
#     # plt.title('Color coded scatter plots of size counts')
#     plt.grid(axis='y', linestyle='--', alpha=0.3)
#     plt.savefig(plotsavepath + '/' + 'sizeHistogramNotBucket.png')
#
#     plt.ylim(0, p99)
#     plt.savefig(plotsavepath + '/' + 'sizeHistogram_p99.png')


def all_analysis_app(trackedimagepath, segPath, origImgPath, t1=None, t2=None):
    from sizeQuartileAnalysis import getEventIntensityPlotsTest
    segParams = pd.read_csv(segPath + '/segmentation_parameters.csv')
    sT = segParams['startTime'][0]
    dbpath = os.path.dirname(trackedimagepath)

    ppath = '/Analysis'
    if t1 is not None:
        ppath += str(t1) + '-'
    if t2 is not None:
        ppath += str(t2)

    plotsavepath = dbpath + ppath
    if not os.path.isdir(plotsavepath):
        os.makedirs(plotsavepath)
    getHistogramUno(dbpath, plotsavepath, t1=t1, t2=t2)
    getEventIntensityPlotsTest(plotsavepath, sT, trackedimagepath,origImgPath, t1=t1, t2=t2)
    getIntensityChangeContribution(dbpath,sT, origImgPath, plotsavepath, t1=t1, t2=t2)
    getIntensityChange(dbpath, sT, origImgPath,plotsavepath, t1=t1, t2=t2)
    getIntensityChangeAvg(dbpath, sT, origImgPath,plotsavepath, t1=t1, t2=t2)
    # getHistogram(trackedimagepath,plotsavepath, t1=t1, t2=t2)
    # getHistogramUno(dbpath, plotsavepath, t1=t1, t2=t2)
    print('\n','#'*30, '\n')
    print('DONE!!!')
