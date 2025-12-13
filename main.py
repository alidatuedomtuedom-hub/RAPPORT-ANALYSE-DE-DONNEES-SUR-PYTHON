#coding:utf8

import pandas as pd
import matplotlib.pyplot as plt

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/
with open("data/resultats-elections-presidentielles-2022-1er-tour.csv",encoding="utf-8") as fichier:
    contenu = pd.read_csv(fichier)
print(contenu)
pd.DataFrame(contenu)
# Mettre dans un commentaire le numéro de la question
# Question 1
# ...
#Question 6

nb_lignes= len(contenu)
nb_colonnes=len(contenu.columns)

print("nombre de lignes:", nb_lignes)
print("nombe de colonnes:", nb_colonnes)

#Question 7
print(contenu.dtypes)

#Question 8

print(contenu.head())
print(contenu.columns)

#Quetion 9
inscrit=contenu["Inscrits"]
print(inscrit)

#Question 10

total_inscrits=inscrit.sum()
print("nombre total des inscrits",total_inscrits)
somme_colonnes=[]
for col in contenu.columns:
    if contenu[col].dtype in ["int64", "float64"]:
        somme_colonnes.append(contenu[col].sum())

print(somme_colonnes)

for col in contenu.columns:
    if contenu[col].dtype in ["int64", "float64"]:
        print(col, ":", contenu[col].sum())
    
#Question 11 

for i in range(len(contenu)):
    dept=contenu.loc[i, "Libellé du département"]
    inscrits = contenu.loc[i, "Inscrits"]
    votants=contenu.loc[i, "Votants"]
    plt.figure(figsize=(8,6))
    plt.bar(["Inscrits", "Votants"], [inscrits, votants],color= ['blue', 'red'])
    plt.title(f"{dept}")
    plt.ylabel("Nombre de personne")
    plt.ticklabel_format(style='plain', axis='y')
    plt.savefig(f"{dept}.png")
    plt.close()

#Question 12
import os
os.makedirs("images_circulaires", exist_ok=True)

for i in range(len(contenu)):
    dept=contenu.loc[i, "Libellé du département"]
    blancs=contenu.loc[i, "Blancs"]
    nuls =contenu.loc[i, "Nuls"]
    votants = contenu.loc[i, "Votants"]
    abstentions = contenu.loc[i, "Abstentions"]
    exprimés = votants - blancs - nuls
    valeurs = [blancs, nuls, exprimés, abstentions]
    labels =["Blancs", "Nuls", "Exprimés","Absention"]
    couleurs=["lightgrey", "blue", "green", "yellow"]
    plt.figure(figsize=(7,7))
    plt.pie(valeurs, labels=labels, colors=couleurs, autopct='%1.1f%%', startangle=90)
    plt.title(f"{dept}")
    plt.savefig(f"images_circulaires/{dept}.png")
    plt.close()

#Question 13

plt.figure(figsize=(8,5))
plt.hist(contenu["Inscrits"], bins=20, density=True, edgecolor='black', color='green')
plt.title("histogramme du nombre d'inscrits")
plt.xlabel("nombre d'inscrits")
plt.ylabel("Densité")
plt.grid(alpha=0.3)
plt.show()