
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time, random
from datetime import datetime, timedelta

# ======================================================
# STEP 1: SYNTHETIC GST E-WAY BILL GENERATOR
# ======================================================
def generate_ewaybill_data(n=20000, export=True, filename="ewaybills.csv"):
    def random_gstin():
        return ''.join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=15))

    cities = [
        ("Bengaluru", "KA"), ("Mumbai", "MH"), ("Delhi", "DL"),
        ("Chennai", "TN"), ("Hyderabad", "TG"), ("Kolkata", "WB"),
        ("Pune", "MH"), ("Ahmedabad", "GJ"), ("Jaipur", "RJ"),
        ("Lucknow", "UP")
    ]

    data = []
    for i in range(n):
        ewayBillNo = f"EWB{100000+i}"
        genDate = datetime.now() - timedelta(days=random.randint(0, 10))
        validUpto = genDate + timedelta(days=random.randint(1, 5))

        fromPlace, fromState = random.choice(cities)
        toPlace, toState = random.choice(cities)

        row = {
            "ewayBillNo": ewayBillNo,
            "genDate": genDate.strftime("%Y-%m-%d %H:%M:%S"),
            "fromGstin": random_gstin(),
            "toGstin": random_gstin(),
            "fromPlace": fromPlace,
            "toPlace": toPlace,
            "fromState": fromState,
            "toState": toState,
            "transMode": random.choice(["Road", "Rail", "Air", "Ship"]),
            "vehicleNo": f"KA{random.randint(10,99)}AB{random.randint(1000,9999)}",
            "docNo": f"INV{random.randint(1000,9999)}",
            "docDate": (genDate - timedelta(days=random.randint(1,3))).strftime("%Y-%m-%d"),
            "distance": random.randint(50, 2000),
            "validUpto": validUpto.strftime("%Y-%m-%d %H:%M:%S")
        }
        data.append(row)

    df = pd.DataFrame(data)
    if export:
        df.to_csv(filename, index=False)
    return df

# ======================================================
# STEP 2: SUPPLY CHAIN GRAPH
# ======================================================
def build_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_node(row["fromPlace"], state=row["fromState"], type="Source")
        G.add_node(row["toPlace"], state=row["toState"], type="Destination")
        G.add_edge(
            row["fromPlace"],
            row["toPlace"],
            distance=row["distance"],
            mode=row["transMode"],
            ewayBillNo=row["ewayBillNo"]
        )
    return G

def plot_graph(G):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_color="lightblue",
            node_size=1200, font_size=8, arrows=True)
    labels = nx.get_edge_attributes(G, "distance")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=7)
    st.pyplot(plt)

# ======================================================
# STEP 3: EMISSION PREDICTION (Random Forest)
# ======================================================
def train_emission_model(df):
    X = df[["distance"]].copy()
    X["mode_encoded"] = df["transMode"].map({"Road": 1, "Rail": 2, "Air": 3, "Ship": 4})
    y = df["distance"] * np.where(df["transMode"]=="Air", 0.3, 0.13)  # synthetic emissions

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return model, mae, rmse

# ======================================================
# STEP 4: DELAY PREDICTION (GNN)
# ======================================================
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train_delay_gnn(df):
    city_to_idx = {city: idx for idx, city in enumerate(set(df["fromPlace"]).union(df["toPlace"]))}
    df["source"] = df["fromPlace"].map(city_to_idx)
    df["target"] = df["toPlace"].map(city_to_idx)

    x = torch.tensor(df[["distance"]].values, dtype=torch.float)
    y = torch.tensor((df["distance"] > 1000).astype(int).values, dtype=torch.long)
    edge_index = torch.tensor([df["source"].values, df["target"].values], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    model = GNN(input_dim=1, hidden_dim=8, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

    pred = out.argmax(dim=1)
    acc = (pred == data.y).sum().item() / len(y)
    return acc

# ======================================================
# STEP 5: STREAMLIT DASHBOARD
# ======================================================
def main():
    st.title("🚀 Vyaapti – Green Supply Chain Digital Twin")
    st.sidebar.header("Controls")

    n = st.sidebar.slider("Number of Synthetic E-way Bills", 50, 500, 200)

    if st.sidebar.button("Generate & Analyze"):
        df = generate_ewaybill_data(n)
        st.subheader("📂 Sample E-way Bills")
        st.write(df.head())

        st.subheader("📊 Supply Chain Graph")
        G = build_graph(df)
        plot_graph(G)

        # Export graph as nodes.csv + edges.csv
        nodes = pd.DataFrame([
            {"city": node, "state": data.get("state"), "type": data.get("type")}
            for node, data in G.nodes(data=True)
        ])
        nodes.to_csv("nodes.csv", index=False)

        edges = pd.DataFrame([
            {"from": u, "to": v, "distance": data.get("distance"),
             "mode": data.get("mode"), "ewayBillNo": data.get("ewayBillNo")}
            for u, v, data in G.edges(data=True)
        ])
        edges.to_csv("edges.csv", index=False)

        model, mae, rmse = train_emission_model(df)
        acc = train_delay_gnn(df)

        st.subheader("🌱 Emission Model Performance")
        st.write(f"MAE: {mae:.2f} | RMSE: {rmse:.2f}")

        st.subheader("🚚 Delay Prediction Accuracy (GNN)")
        st.write(f"{acc:.2%}")

        st.subheader("📦 Live Shipment Predictions")
        shipment_log = st.empty()
        for i in range(5):
            new_row = df.sample(1).iloc[0].copy()
            new_row["pred_emission"] = model.predict([[new_row["distance"],
                                                       {"Road":1,"Rail":2,"Air":3,"Ship":4}[new_row["transMode"]]]])[0]
            shipment_log.write(pd.DataFrame([new_row]))
            time.sleep(2)

if __name__ == "__main__":
    main()
