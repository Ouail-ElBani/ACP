import numpy as np

# Obtenez l'entrée de l'utilisateur pour les dimensions de la matrice de données
try:
    num_rows = int(input("Entrez le nombre de lignes : "))
    num_cols = int(input("Entrez le nombre de colonnes : "))
except ValueError:
    print("Saisie non valide. Veuillez entrer des valeurs numériques.")
    exit()

# Initialisez une matrice de données vide
data = np.zeros((num_rows, num_cols))

# Obtenez l'entrée de l'utilisateur pour chaque valeur de la matrice de données
for i in range(num_rows):
    for j in range(num_cols):
        try:
            data[i][j] = float(input(f"Entrez la valeur pour la ligne {i + 1}, colonne {j + 1} : "))
        except ValueError:
            print("Saisie non valide. Veuillez entrer des valeurs numériques.")
            exit()

# Calculez la matrice centrée en soustrayant la moyenne de chaque colonne
mean = np.mean(data, axis=0)
centered_data = data - mean

# Calculez la matrice centrée réduite ()
reduced_centered_data = centered_data / np.std(centered_data, axis=0)

# Calculez la matrice de covariance
covariance_matrix = np.cov(centered_data, rowvar=False)

# matrice de corrélation
correlation_matrix = np.corrcoef(data, rowvar=False)


# Calculez les valeurs propres et les vecteurs propres de la matrice de covariance
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Trier les valeurs propres et les vecteurs propres en ordre décroissant
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

principal_components = np.dot(centered_data, eigenvectors)
# Affichez les résultats
print("\nMatrice de Données :")
print(data)
print("\nMatrice Centrée :")
print(centered_data)
print("\nMatrice Centrée Réduite :")
print(reduced_centered_data)
print("\nMatrice de Covariance :")
print(covariance_matrix)
print("\nMatrice de Corrélation :")
print(correlation_matrix)
print("\nValeurs Propres :")
print(eigenvalues)
print("\nVecteurs Propres (Axes Principaux) :")
print(eigenvectors)
print("\nNouvelles Composantes :")
print(principal_components)