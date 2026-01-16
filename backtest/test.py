import pandas as pd

df = pd.read_csv("optimization_results.csv")

# adjust these names after you paste the header if needed
return_col = [c for c in df.columns if "return" in c.lower() or "profit" in c.lower() or "net" in c.lower()][0]
dd_cols = [c for c in df.columns if "drawdown" in c.lower() or "dd" in c.lower()]

print("Return col:", return_col)
print("DD cols:", dd_cols)

# show top results by return
print(df.sort_values(return_col, ascending=False).head(15).to_string(index=False))
