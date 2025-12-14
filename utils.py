#imports
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib import cm

from sklearn.metrics import silhouette_score

#Load and process data
def load_input(csv_path, adventurous_weight=0.4):
    df = pd.read_csv(csv_path)
    

    mood_cols = [
        "adventurous", "challenging", "dark","emotional", "funny", "hopeful", "informative", "inspiring", 
        "lighthearted", "mysterious", "reflective", "relaxing", "sad", "tense"]
    df["fuzzy_pace"] = df[["paceslow", "pacemedium", "pacefast"]].idxmax(axis=1)
    df["fuzzy_pace"] = df["fuzzy_pace"].map({"paceslow":"slow", "pacemedium":"medium","pacefast":"fast"})
    df["fuzzy_len"] =  pd.cut(df["length"], bins=[0,300,499,np.inf], labels=["short","medium","long"])

    features = df[mood_cols].copy() #-> takes only moods
    features["adventurous"] *= adventurous_weight
    features_names = list(features.columns)
    
    #normalize data
    scaler = StandardScaler()
    normalized_df = scaler.fit_transform(features)

    return df, normalized_df, scaler, features_names

# train fuzzy clustering
def build_fuzzy_model(normalized_data, n_clusters=4):
    data_T = normalized_data.T

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data_T,
        c=n_clusters,
        m=1.5,
        error=0.005,
        maxiter=1000,
        init=None
    )
    return cntr, u, fpc

# find best number of clusters (without plotting, used in the user version)
def evaluate_fpc_without_analysis(normalized_df, k_min,k_max):
    fpcs = []
    silhouettes = []
    ks = range(k_min,k_max+1)
    for k in ks:
        _, u, fpc = build_fuzzy_model(normalized_df,n_clusters=k)
        labels = np.argmax(u,axis=0)
        fpcs.append(fpc)
        silhouettes.append(silhouette_score(normalized_df,labels))
    
    best_k_fpc = ks[np.argmax(fpcs)]
    best_k_silouhette = ks[np.argmax(silhouettes)]

    #NOTE: choose methode to return best k number!!!!!
    return best_k_fpc

# find best nbr of clusters (with plotting)
def evaluate_fpc(normalized_df, k_min,k_max):
    fpcs = []
    silhouettes = []
    ks = range(k_min,k_max+1)
    for k in ks:
        _, u, fpc = build_fuzzy_model(normalized_df,n_clusters=k)
        labels = np.argmax(u,axis=0)
        fpcs.append(fpc)
        silhouettes.append(silhouette_score(normalized_df,labels))
    #plot metrics
    plt.figure(figsize=(10,4))
    plt.plot(ks,fpcs,marker="o",label="FPC")
    plt.plot(ks,silhouettes,marker='x')
    plt.xlabel("Number of clusters")
    plt.ylabel("Score")
    plt.title("Number of cluster analysis")
    
    best_k_fpc = ks[np.argmax(fpcs)]
    best_k_silouhette = ks[np.argmax(silhouettes)]
    print("Best Number cluster (fpc) = ", best_k_fpc)
    print("Best Number cluster (silouhette) = ", best_k_silouhette)
    #NOTE: choose methode to return best k number!!!!!
    return best_k_fpc

# plot the clusters 
def visualize_clusters(normalized_data, memberships, n_clusters, feature):
    reducer = umap.UMAP(random_state=42)
    emb = reducer.fit_transform(normalized_data)
    cluster_labels = np.argmax(memberships,axis=0)
    fig = px.scatter(
        x=emb[:,0],
        y=emb[:,1],
        color=cluster_labels.astype(str),
        hover_data={
            "title":feature["title"],
            "index":feature.index,
        },
        title="Umpa projection of books (colored by cluster)"
    )
    fig.show()

# plot and print where cluster centers are situated (in relation to moods)
def plot_cluster_centers(cntr, feature_names, scaler, top_n):
    colors=[]
    for f in feature_names:
        if f.startswith("len"):
            colors.append("red")
        elif f.startswith("pace"):
            colors.append("green")
        else:
            colors.append("blue")

    plt.figure(figsize=(14,6))
    for i, center in enumerate(cntr):
        plt.plot(center, label=f"Cluster {i}",linewidth=2)
        if scaler:
            center_orig = scaler.inverse_transform(center.reshape(1,-1))[0]
        else:
            center_orig = center
        top_idx = center_orig.argsort()[::-1][:top_n]
        top_features = [feature_names[j] for j in top_idx]
        print(f"Cluster {i} top features: {top_features}")
    for idx, f in enumerate(feature_names):
        plt.axvline(x=idx, color=colors[idx],alpha=0.1)
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.title("Cluster centers(feature importance)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# convert user input into feature vector
def encode_user_input(user_input, feature_names, adventurous_weight=0.4):
    title,author,length,mood_values,pace = user_input
    mood_dict = dict(zip(["adventurous", "challenging", "dark", "emotional", "funny", "hopeful", "informative",
                          "inspiring", "lighthearted", "mysterious", "reflective", "relaxing", "sad", "tense"], mood_values))
    mood_vec = np.array([mood_dict[f] * (adventurous_weight  if  f == "adventurous" else 1.0) for f in feature_names])

    return mood_vec.reshape(1,-1)

# normalising/scaling user input
def scale_user_input(feature_vec, scaler, feature_name):
    df = pd.DataFrame(feature_vec, columns=feature_name)
    return scaler.transform(df)

# predicting in which cluster the user input is situatied
def predict_cluster(user_input, centers):
    data_T = user_input.T
    u,u0,d,jm,p,fpc = fuzz.cluster.cmeans_predict(
        test_data=data_T,
        cntr_trained=centers,
        m=1.5,
        error=0.005,
        maxiter=1000
    )
    cluster_index = np.argmax(u,axis=0)[0]
    return cluster_index, u

# apply fuzzy rules
def apply_fuzzy_filters(df, user_len, user_pace, book_pref_ctrl):
    book_pref_sim = ctrl.ControlSystemSimulation(book_pref_ctrl)

    prefs = []
    length_map = {"short":0, "medium":1, "long":2}
    pace_map = {"slow":0, "medium":1, "fast":2}

    for _, row in df.iterrows():
        if user_len is not None:
            book_pref_sim.input["book_len"] = length_map[row["fuzzy_len"]]
        else:
            # default in case it's None
            book_pref_sim.input["book_len"] = 1
        if user_pace is not None: 
            book_pref_sim.input["book_pace"] = pace_map[row["fuzzy_pace"]]
        else:
            # default in case it's None
            book_pref_sim.input["book_pace"] = 1
        book_pref_sim.compute()
        prefs.append(book_pref_sim.output["preference"])
    df["fuzzy_preference"] = prefs
    return df

# build dynamic fuzzy rules so that it is user dependant (based on lenght and pace of books)
def dynamic_rules(book_len, book_pace, user_len, user_pace, preference):
    rules = []
    for l in ["short", "medium", "long"]:
        for p in ["slow", "medium", "fast"]:
            # exact match -> high, one match -> medium, no match -> low
            if l == user_len and p == user_pace:
                out = preference["high"]
            elif l == user_len or p == user_pace:
                out = preference["medium"]
            else:
                out = preference["low"]
            rules.append(ctrl.Rule(book_len[l] & book_pace[p], out))
    return rules

# recommend 10 books (4 options: 1. just using clustering 2. clustering + fuzzy rules 3. clustering + ratings 4. all together)
def recommend_books(
    df,
    memberships,
    cluster_index,
    book_pref_ctrl,
    top_k=10,
    user_len=None,
    user_pace=None,
    rule_weight=0.3,
    use_rules=True,
    use_rating=False,
):
    df = df.copy()
    cluster_strenght = memberships[cluster_index]
    df["cluster_strenght"] = cluster_strenght

    # add ratings
    if use_rating:
        if "rating" in df.columns:
            rating_norm = pd.to_numeric(df["rating"], errors="coerce").fillna(1.0) / 5.0
            rating_norm = rating_norm.clip(lower=0.0, upper=1.0)
            df["cluster_strenght"] = df["cluster_strenght"] * rating_norm.values

    # add fuzzy len-pace rules
    if use_rules:
        df = apply_fuzzy_filters(df, user_len, user_pace, book_pref_ctrl)
        df["cluster_norm"] = (df["cluster_strenght"] - df["cluster_strenght"].min())/(df["cluster_strenght"].max()- df["cluster_strenght"].min())
        df["fuzzy_norm"] = (df["fuzzy_preference"]-df["fuzzy_preference"].min())/(df["fuzzy_preference"].max()-df["fuzzy_preference"].min())
        df["cluster_strenght"] = (1-rule_weight) * df["cluster_norm"] + rule_weight * df["fuzzy_norm"]
    return df.sort_values("cluster_strenght", ascending=False).head(top_k)

# whole pipeline (without graph analysis) for the user-experiences notebook (Bookwyrm.ipynb)
def recs_pipeline(user_input):
    df_raw, normalized_df, scaler, feat_names = load_input("./book_list.csv", adventurous_weight=0)
    # training
    centers, memberships, fpc = build_fuzzy_model(normalized_df, n_clusters=4)

    # input fuzzy variables
    book_len = ctrl.Antecedent(np.arange(0,3,1), "book_len")
    book_pace = ctrl.Antecedent(np.arange(0,3,1), "book_pace")

    # output fuzzy variable
    preference = ctrl.Consequent(np.arange(0,11,1), "preference")

    # membership functions
    book_len["short"] = fuzz.trimf(book_len.universe, [0,0,1])
    book_len["medium"] = fuzz.trimf(book_len.universe, [0,1,2])
    book_len["long"] = fuzz.trimf(book_len.universe, [1,2,2])

    book_pace["slow"] = fuzz.trimf(book_pace.universe, [0,0,1])
    book_pace["medium"] = fuzz.trimf(book_pace.universe, [0,1,2])
    book_pace["fast"] = fuzz.trimf(book_pace.universe, [1,2,2])

    preference["low"] = fuzz.trimf(preference.universe, [0,0,5])
    preference["medium"] = fuzz.trimf(preference.universe, [0,5,10])
    preference["high"] = fuzz.trimf(preference.universe, [5,10,10])

    if user_input[5] > 0:
        best_k = user_input[5]
    else:
        best_k = evaluate_fpc_without_analysis(normalized_df, k_min=6, k_max=12)

    centers,memberships, fpc = build_fuzzy_model(normalized_df, best_k)
    
    user = user_input[:-1]
    user_len = user[2]
    user_pace = user[4]
    fv = encode_user_input(user, feat_names, adventurous_weight=0)
    fv_scaled = scale_user_input(fv,scaler, feat_names)
    cluster_index, u = predict_cluster(fv_scaled, centers)

    # computing dynamic fuzzy rules (lenght, pace)
    rules = dynamic_rules(book_len, book_pace, user_len, user_pace, preference)
    book_pref_ctrl = ctrl.ControlSystem(rules)

    # OPTION 1: get recommendations using only the mood-based fuzzy clustering
    recs_only_clustering = recommend_books(
        df_raw, memberships, cluster_index, book_pref_ctrl,
        top_k=10, user_len=user_len, user_pace=user_pace,
        use_rules=False, use_rating=False,
    )

    # OPTION 2: get recommendation while adding dynamic fuzzy rules on top of clustering 
    recs_with_rules = recommend_books(
        df_raw, memberships, cluster_index, book_pref_ctrl,
        top_k=10, user_len=user_len, user_pace=user_pace,
        rule_weight=0.4, use_rules=True, use_rating=False,
    )

   # OPTION 3: get recommendations using the ratings on top of clustering
    recs_with_rating = recommend_books(
        df_raw, memberships, cluster_index, book_pref_ctrl,
        top_k=10, user_len=user_len, user_pace=user_pace,
        use_rules=False, use_rating=True,
    )

    # OPTION 4: get the recommendations using fuzzy rules and ratings on top of clustering
    recs_with_rules_nd_ratings = recommend_books(
        df_raw, memberships, cluster_index, book_pref_ctrl,
        top_k=10, user_len=user_len, user_pace=user_pace,
        rule_weight=0.4, use_rules=True, use_rating=True,
    )
 
    return recs_only_clustering, recs_with_rules, recs_with_rating, recs_with_rules_nd_ratings
