"""Génère les deux notebooks du Bloc 3 (ML et DL) à partir de cellules définies ici."""
import nbformat as nbf


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[3:].replace("$", r"\$")) if c.startswith("MD:") else nbf.v4.new_code_cell(c) for c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return n


COMMON_PREP = r'''import re, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
SEED = 42
np.random.seed(SEED)

df = pd.read_csv("data/luxury_watch.csv", encoding="utf-8-sig")
df["Price (USD)"] = pd.to_numeric(df["Price (USD)"].astype(str).str.replace(",", ""), errors="coerce")
df = df.dropna(subset=["Price (USD)"]).reset_index(drop=True)
df["Complications"] = df["Complications"].fillna("None")
df["Power Reserve"] = df["Power Reserve"].fillna("Unknown")
NUM = ["Case Diameter (mm)", "Case Thickness (mm)", "Band Width (mm)"]
CAT = ["Brand", "Model", "Case Material", "Strap Material", "Movement Type", "Water Resistance",
       "Dial Color", "Crystal Material", "Complications", "Power Reserve"]
TARGET = "Price (USD)"
SEUIL = 8400  # médiane arrondie : frontière Accessible / Premium
df["Premium"] = (df[TARGET] >= SEUIL).astype(int)
print(df.shape)'''

ML = [
    "MD:# Bloc 3 — Notebook 1 : Machine Learning\n## Analyse prédictive des prix des montres de luxe\n\n**Auteurs du mini-projet :** Ronny Douglas, Victorien Alleg — MSc 2025-2026, Eugenia School  \n**Cours :** Machine Learning & Deep Learning (Georgina)\n\n**Objectif.** Comprendre les facteurs qui déterminent le prix d'une montre de luxe et construire deux modèles :\n1. une **régression** qui prédit le prix (USD) ;\n2. une **classification** qui range une montre en *Accessible* (< 8 400 $) ou *Premium* (≥ 8 400 $).\n\n**Données.** *Luxury Watch Prices* (Kaggle) : 507 montres, 14 variables (marque, modèle, matériaux, mouvement, étanchéité, dimensions, complications, réserve de marche, prix).",
    "MD:## 1. Chargement et nettoyage\nLe prix est stocké en texte (`\"9,500\"`) : on le convertit en nombre. Une ligne sans prix est retirée. Les valeurs manquantes de *Complications* (122) et *Power Reserve* (14) sont des absences réelles : on les code comme catégorie (`None` / `Unknown`) plutôt que de les supprimer.",
    COMMON_PREP,
    "df.head()",
    "df[NUM + [TARGET]].describe().round(1)",
    "MD:## 2. Analyse exploratoire\n### 2.1 Distribution des prix",
    r'''fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(df[TARGET], bins=30, kde=True, ax=ax)
ax.set_title("Distribution des prix des montres de luxe"); ax.set_xlabel("Prix ($)"); ax.set_ylabel("Fréquence")
plt.savefig("figs/01_distribution_prix.png", dpi=150, bbox_inches="tight"); plt.show()
print(f"Médiane : {df[TARGET].median():,.0f} $  |  Moyenne : {df[TARGET].mean():,.0f} $  |  Asymétrie : {df[TARGET].skew():.2f}")''',
    "MD:Forte asymétrie à droite : la majorité des montres se situe entre 5 000 et 15 000 $, avec une longue traîne ultra-premium (jusqu'à 70 000 $).\n\n### 2.2 Prix moyen par marque",
    r'''pm = df.groupby("Brand")[TARGET].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 4.5))
pm.plot.bar(ax=ax, color="#2b6cb0")
ax.set_title("Prix moyen par marque"); ax.set_ylabel("Prix moyen ($)"); ax.set_xlabel("")
plt.savefig("figs/02_prix_marque.png", dpi=150, bbox_inches="tight"); plt.show()
pm.head(5).round(0)''',
    "MD:### 2.3 Corrélations — variables numériques",
    r'''corr = df[NUM + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Matrice de corrélation")
plt.savefig("figs/03_correlation.png", dpi=150, bbox_inches="tight"); plt.show()''',
    "MD:Les dimensions sont corrélées entre elles mais **négativement ou faiblement** avec le prix : dans le luxe, « plus grand » ne veut pas dire « plus cher ». L'épaisseur est la plus liée (négativement) : les montres haut de gamme privilégient la finesse.\n\n### 2.4 Variables catégorielles — V de Cramér",
    r'''from scipy.stats import chi2_contingency
def cramers_v(x, y):
    t = pd.crosstab(x, y); chi2 = chi2_contingency(t)[0]; n = t.values.sum(); r, k = t.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))
price_bin = pd.qcut(df[TARGET], 5, labels=False)
cv = pd.Series({c: cramers_v(df[c], price_bin) for c in CAT}).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 4)); cv.plot.bar(ax=ax, color="#2f855a")
ax.set_title("Association avec le prix (V de Cramér, prix en quintiles)"); ax.set_ylabel("V de Cramér")
plt.savefig("figs/04_cramer.png", dpi=150, bbox_inches="tight"); plt.show(); cv.round(2)''',
    "MD:### 2.5 Impact du matériau du boîtier",
    r'''order = df.groupby("Case Material")[TARGET].median().sort_values().index
fig, ax = plt.subplots(figsize=(12, 4.5))
sns.boxplot(data=df, x="Case Material", y=TARGET, order=order, ax=ax, color="#90cdf4")
ax.tick_params(axis="x", rotation=60); ax.set_title("Prix par matériau du boîtier"); ax.set_xlabel("")
plt.savefig("figs/05_materiau.png", dpi=150, bbox_inches="tight"); plt.show()''',
    "MD:Les métaux précieux (or rose, or blanc, platine) dominent l'échelle des prix ; l'acier concentre l'essentiel des montres et une forte dispersion.\n\n## 3. Préparation pour la modélisation\nEncodage *one-hot* des variables catégorielles, découpage **80 % entraînement / 20 % test** stratifié sur la classe Premium.",
    r'''from sklearn.model_selection import train_test_split
X = pd.get_dummies(df[NUM + CAT], columns=CAT, drop_first=False).astype(float)
y_reg, y_clf = df[TARGET], df["Premium"]
X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=SEED, stratify=y_clf)
print("Train :", X_tr.shape, " Test :", X_te.shape)''',
    "MD:## 4. Régression du prix\n### 4.1 Régression linéaire simple (diamètre seul)",
    r'''from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
res = {}
def evaluer(nom, y_true, y_pred):
    res[nom] = {"R²": r2_score(y_true, y_pred), "RMSE": mean_squared_error(y_true, y_pred) ** 0.5, "MAE": mean_absolute_error(y_true, y_pred)}
    return res[nom]
lr_s = LinearRegression().fit(X_tr[["Case Diameter (mm)"]], yr_tr)
p_s = lr_s.predict(X_te[["Case Diameter (mm)"]])
print(evaluer("Régression simple", yr_te, p_s))
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(X_te["Case Diameter (mm)"], yr_te, alpha=.6, label="Réel")
xs = np.linspace(X_te["Case Diameter (mm)"].min(), X_te["Case Diameter (mm)"].max(), 50)
ax.plot(xs, lr_s.predict(pd.DataFrame({"Case Diameter (mm)": xs})), "r", label="Droite de régression")
ax.set_xlabel("Diamètre (mm)"); ax.set_ylabel("Prix ($)"); ax.legend(); ax.set_title("Régression simple : prix ~ diamètre")
plt.savefig("figs/06_reg_simple.png", dpi=150, bbox_inches="tight"); plt.show()''',
    "MD:Le diamètre seul n'explique presque rien (R² proche de 0, voire négatif sur le test).\n\n### 4.2 Régression linéaire multiple",
    r'''lr_m = LinearRegression().fit(X_tr, yr_tr)
p_m = lr_m.predict(X_te)
print(evaluer("Régression multiple", yr_te, p_m))
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].scatter(yr_te, p_m, alpha=.6); m = yr_te.max(); axs[0].plot([0, m], [0, m], "r--")
axs[0].set_xlabel("Prix réel ($)"); axs[0].set_ylabel("Prix prédit ($)"); axs[0].set_title("Prédit vs réel")
sns.histplot(yr_te - p_m, bins=30, ax=axs[1]); axs[1].set_title("Distribution des résidus")
plt.savefig("figs/07_reg_multiple.png", dpi=150, bbox_inches="tight"); plt.show()''',
    "MD:### 4.3 Random Forest",
    r'''rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X_tr, yr_tr)
p_rf = rf.predict(X_te)
print(evaluer("Random Forest", yr_te, p_rf))
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(yr_te, p_rf, alpha=.6, color="#2f855a"); ax.plot([0, m], [0, m], "r--")
ax.set_xlabel("Prix réel ($)"); ax.set_ylabel("Prix prédit ($)"); ax.set_title("Random Forest : prédit vs réel")
plt.savefig("figs/08_rf.png", dpi=150, bbox_inches="tight"); plt.show()''',
    "MD:### 4.4 Comparaison des modèles et variables importantes",
    r'''comp = pd.DataFrame(res).T.round(3); display(comp)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
comp["RMSE"].plot.bar(ax=axs[0], color="#90cdf4", title="RMSE ($) — plus bas = mieux")
comp["R²"].plot.bar(ax=axs[1], color="#f56565", title="R² — plus haut = mieux")
for a in axs: a.tick_params(axis="x", rotation=20)
plt.savefig("figs/09_comparaison.png", dpi=150, bbox_inches="tight"); plt.show()
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8, 4)); imp[::-1].plot.barh(ax=ax, color="#2b6cb0")
ax.set_title("Top 10 des variables (Random Forest)")
plt.savefig("figs/10_importance.png", dpi=150, bbox_inches="tight"); plt.show()''',
    "MD:### 4.5 Validation croisée (robustesse)\nUn seul découpage train/test peut être chanceux : on vérifie le R² du Random Forest en validation croisée à 5 plis.",
    r'''from sklearn.model_selection import cross_val_score, KFold
cv_r2 = cross_val_score(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1), X, y_reg, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2")
print("R² par pli :", cv_r2.round(3), " moyenne :", cv_r2.mean().round(3), "±", cv_r2.std().round(3))''',
    "MD:## 5. Classification Accessible / Premium — régression logistique",
    r'''from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(X_tr, yc_tr)
pc = logit.predict(X_te); proba = logit.predict_proba(X_te)[:, 1]
print(f"Accuracy : {accuracy_score(yc_te, pc):.3f}  |  ROC-AUC : {roc_auc_score(yc_te, proba):.3f}")
print(classification_report(yc_te, pc, target_names=["Accessible", "Premium"]))
fig, ax = plt.subplots(figsize=(4.5, 4))
sns.heatmap(confusion_matrix(yc_te, pc), annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Accessible", "Premium"], yticklabels=["Accessible", "Premium"])
ax.set_xlabel("Prédit"); ax.set_ylabel("Réel"); ax.set_title("Matrice de confusion")
plt.savefig("figs/11_confusion.png", dpi=150, bbox_inches="tight"); plt.show()''',
    "MD:## 6. Prédictions sur trois montres de référence\nPrix prédit par le Random Forest (intervalle ± RMSE) et probabilité de classe par la régression logistique.",
    r'''refs = {"Rolex Submariner": ("Rolex", "Submariner"), "Omega Seamaster": ("Omega", "Seamaster"), "Patek Philippe Nautilus": ("Patek Philippe", "Nautilus")}
rmse_rf = res["Random Forest"]["RMSE"]; rows = []
for nom, (b, mo) in refs.items():
    x = X[(df.Brand == b) & (df.Model == mo)].iloc[[0]]
    prix = rf.predict(x)[0]; pp = logit.predict_proba(x)[0, 1]
    rows.append({"Montre": nom, "Prix prédit ($)": round(prix), "Intervalle bas": round(prix - rmse_rf), "Intervalle haut": round(prix + rmse_rf),
                 "Classe": "Premium" if pp >= .5 else "Accessible", "Probabilité de la classe": round(max(pp, 1 - pp), 3)})
pred = pd.DataFrame(rows); display(pred)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(pred["Montre"], pred["Prix prédit ($)"], yerr=rmse_rf, capsize=6, color=["#48bb78", "#4299e1", "#9f7aea"])
ax.axhline(SEUIL, ls="--", color="r", label="Seuil 8 400 $"); ax.legend(); ax.set_ylabel("Prix prédit ($)")
ax.set_title("Prix prédits (± RMSE)")
plt.savefig("figs/12_predictions.png", dpi=150, bbox_inches="tight"); plt.show()''',
    "MD:## 7. Conclusions\n- **Random Forest** est le meilleur modèle de prix : il capte les effets non linéaires (marque × modèle × matériau) que la régression linéaire rate.\n- La **régression simple** sur le diamètre est inutilisable : la taille ne fait pas le prix.\n- La **régression logistique** classe correctement la grande majorité des montres en Accessible / Premium ; les erreurs se concentrent près du seuil de 8 400 $.\n- **Facteurs clés** : modèle et marque (Nautilus, Royal Oak, Audemars Piguet…), matériau du boîtier, étanchéité.\n- **Limites** : 506 montres seulement, prix catalogue (pas de marché secondaire), modèles dupliqués dans le jeu → le test contient des références déjà vues, ce qui flatte les scores. Pistes : historique d'enchères, état, année.",
]

DL = [
    "MD:# Bloc 3 — Notebook 2 : Deep Learning\n## Réseau de neurones (MLP) pour le prix et le segment des montres de luxe\n\n**Auteurs du mini-projet :** Ronny Douglas, Victorien Alleg — MSc 2025-2026, Eugenia School  \n**Cours :** Machine Learning & Deep Learning (Georgina)\n\n**Question.** Un réseau de neurones fait-il mieux que le Random Forest du notebook 1 sur ce jeu de données tabulaire ?\n\nOn construit avec **PyTorch** un perceptron multicouche (MLP) :\n- **tête régression** : prédit le log du prix ;\n- **tête classification** : prédit Accessible / Premium.\n\nMêmes données, même découpage 80/20 (graine 42) que le notebook 1 pour une comparaison honnête.",
    "MD:## 1. Données (identiques au notebook 1)",
    COMMON_PREP,
    r'''import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
torch.manual_seed(SEED)
X = pd.get_dummies(df[NUM + CAT], columns=CAT).astype(float)
y_reg, y_clf = df[TARGET], df["Premium"]
X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=SEED, stratify=y_clf)
# 15 % du train sert de validation pour l'arrêt anticipé (early stopping)
X_tr2, X_va, yr_tr2, yr_va, yc_tr2, yc_va = train_test_split(X_tr, yr_tr, yc_tr, test_size=0.15, random_state=SEED, stratify=yc_tr)
sc = StandardScaler().fit(X_tr2[NUM])
def prep(Xd):
    Xd = Xd.copy(); Xd[NUM] = sc.transform(Xd[NUM]); return torch.tensor(Xd.values, dtype=torch.float32)
Xt, Xv, Xs = prep(X_tr2), prep(X_va), prep(X_te)
print("Entrées :", Xt.shape[1], "variables | train", len(Xt), "| val", len(Xv), "| test", len(Xs))''',
    "MD:## 2. Architecture\nMLP à 2 couches cachées (128 → 64), activation ReLU, *dropout* 0,2 contre le sur-apprentissage, optimiseur Adam avec *weight decay* (régularisation L2). La régression apprend le **log du prix** pour atténuer la longue traîne.",
    r'''class MLP(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, d_out))
    def forward(self, x): return self.net(x)

def entrainer(model, loss_fn, Xtr, ytr, Xva, yva, epochs=600, lr=1e-3, patience=40):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    hist, best, best_state, wait = {"train": [], "val": []}, 1e9, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i + 32]; opt.zero_grad(); l = loss_fn(model(Xtr[idx]), ytr[idx]); l.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lt, lv = loss_fn(model(Xtr), ytr).item(), loss_fn(model(Xva), yva).item()
        hist["train"].append(lt); hist["val"].append(lv)
        if lv < best - 1e-5: best, best_state, wait = lv, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state); return hist
print(MLP(Xt.shape[1], 1))''',
    "MD:## 3. Régression du prix",
    r'''from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
mu, sd = np.log(yr_tr2).mean(), np.log(yr_tr2).std()
ylog = lambda s: torch.tensor(((np.log(s.values) - mu) / sd)[:, None], dtype=torch.float32)
reg = MLP(Xt.shape[1], 1)
h_reg = entrainer(reg, nn.MSELoss(), Xt, ylog(yr_tr2), Xv, ylog(yr_va))
with torch.no_grad(): p_mlp = np.exp(reg(Xs).numpy().ravel() * sd + mu)
r_mlp = {"R²": r2_score(yr_te, p_mlp), "RMSE": mean_squared_error(yr_te, p_mlp) ** .5, "MAE": mean_absolute_error(yr_te, p_mlp)}
print("Époques :", len(h_reg["train"]), "|", {k: round(v, 3) for k, v in r_mlp.items()})
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(h_reg["train"], label="train"); axs[0].plot(h_reg["val"], label="validation"); axs[0].set_title("Courbe d'apprentissage (MSE, log-prix normalisé)"); axs[0].set_xlabel("Époque"); axs[0].legend()
m = yr_te.max(); axs[1].scatter(yr_te, p_mlp, alpha=.6, color="#805ad5"); axs[1].plot([0, m], [0, m], "r--"); axs[1].set_xlabel("Prix réel ($)"); axs[1].set_ylabel("Prix prédit ($)"); axs[1].set_title("MLP : prédit vs réel")
plt.savefig("figs/13_mlp_regression.png", dpi=150, bbox_inches="tight"); plt.show()''',
    "MD:## 4. Classification Accessible / Premium",
    r'''from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
yt = lambda s: torch.tensor(s.values[:, None], dtype=torch.float32)
clf = MLP(Xt.shape[1], 1)
h_clf = entrainer(clf, nn.BCEWithLogitsLoss(), Xt, yt(yc_tr2), Xv, yt(yc_va))
with torch.no_grad(): pr = torch.sigmoid(clf(Xs)).numpy().ravel()
pc = (pr >= .5).astype(int)
print(f"Époques : {len(h_clf['train'])} | Accuracy : {accuracy_score(yc_te, pc):.3f} | ROC-AUC : {roc_auc_score(yc_te, pr):.3f}")
print(classification_report(yc_te, pc, target_names=["Accessible", "Premium"]))
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot(h_clf["train"], label="train"); axs[0].plot(h_clf["val"], label="validation"); axs[0].set_title("Courbe d'apprentissage (entropie croisée)"); axs[0].set_xlabel("Époque"); axs[0].legend()
sns.heatmap(confusion_matrix(yc_te, pc), annot=True, fmt="d", cmap="Purples", ax=axs[1], xticklabels=["Accessible", "Premium"], yticklabels=["Accessible", "Premium"])
axs[1].set_xlabel("Prédit"); axs[1].set_ylabel("Réel"); axs[1].set_title("Matrice de confusion — MLP")
plt.savefig("figs/14_mlp_classification.png", dpi=150, bbox_inches="tight"); plt.show()''',
    "MD:## 5. Comparaison avec le Machine Learning classique",
    r'''from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X_tr, yr_tr); p_rf = rf.predict(X_te)
lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, yc_tr)
tab = pd.DataFrame({
    "Random Forest (ML)": {"R² prix": r2_score(yr_te, p_rf), "RMSE prix ($)": mean_squared_error(yr_te, p_rf) ** .5},
    "MLP (DL)": {"R² prix": r_mlp["R²"], "RMSE prix ($)": r_mlp["RMSE"]}}).T
tab["Accuracy segment"] = [accuracy_score(yc_te, lg.predict(X_te)), accuracy_score(yc_te, pc)]
tab.index = ["ML : Random Forest / Rég. logistique", "DL : MLP PyTorch"]
display(tab.round(3))''',
    "MD:## 6. Conclusions\n- Sur **506 lignes tabulaires**, le réseau de neurones **ne surpasse pas nettement** le Random Forest : c'est le résultat attendu. Les modèles à base d'arbres restent la référence sur les petites données tabulaires ; le deep learning brille sur les grands volumes et les données non structurées (images, texte, son).\n- Le MLP demande plus de réglages (normalisation, log du prix, dropout, early stopping) pour un gain faible ou nul ici.\n- **Recommandation** : en production, garder le **Random Forest** (plus simple, plus explicable via l'importance des variables). Le deep learning deviendrait pertinent avec des **photos de montres** (CNN) ou des **descriptions d'annonces** (modèles de langue) pour enrichir la prédiction.",
]

nbf.write(nb(ML), "01_ML_prix_montres.ipynb")
nbf.write(nb(DL), "02_DL_prix_montres.ipynb")
print("ok")
