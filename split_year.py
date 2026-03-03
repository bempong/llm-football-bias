import pandas as pd

def main():
    # Split the data into 2010-2019 and 2020-2026

    df = pd.read_csv("llm_new_generations/output/4o_mini_2010_2026_exp_3k.csv")
    df_2010_2019 = df[df['example_year'].between(2010, 2019)]
    df_2020_2026 = df[df['example_year'].between(2020, 2026)]

    print("Number of rows in 2010-2019:", len(df_2010_2019))
    print("Number of rows in 2020-2026:", len(df_2020_2026))

    df_2010_2019.to_csv("llm_new_generations/output/4o_mini_2010_2019_3k.csv", index=False)
    df_2020_2026.to_csv("llm_new_generations/output/4o_mini_2020_2026_3k.csv", index=False)

if __name__ == "__main__":
    main()