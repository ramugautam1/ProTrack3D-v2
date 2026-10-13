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


def getEventIntensityPlotsTest(plotsavepath, sT, trackedimagepath, origImgPath, t1=None, t2=None):
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
    FlowObj = newDF[(newDF['t'] >= t1) & (newDF['t'] <= t2)]

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
        ax.set_title(titles[i], fontsize=15)
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

    totObj = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]
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

    splitObj = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
    splitObj

    print('---' * 20, splitObj.columns)

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

    mergeObj = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    mergePrimObj = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    mergeSecObj = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    deadObj = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    bornObj = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    intDF = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    totInt = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    expDF = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    expDFnew = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    densityDF = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    intensityIncDF = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    intensityDecDF = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    avgIntensityDF = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    peakIntDF = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    splitCombObj = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    mergeCombObj = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    avgIntObj = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
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

    avgSizeObj = newDF[(newDF['t']  >=  t1) & (newDF['t'] >= t1) & (newDF['t'] <= t2)]  # .drop(columns=['t'])
    avgSizeObj

    #######################################################################

    dfs = [splitObj, mergeObj, mergePrimObj, mergeSecObj, deadObj, bornObj,
           totObj, totInt, intDF, peakIntDF, densityDF,
           avgIntensityDF, expDF, expDFnew, intensityIncDF, intensityDecDF, mergeCombObj, avgIntObj, avgSizeObj
        , splitCombObj
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
              'Merge Fraction, Overall',
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
