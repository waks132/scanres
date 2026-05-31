#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network SVD/PCA Analyzer
==========================
Application d'analyse de réseaux critiques basée sur la Décomposition 
en Valeurs Singulières (SVD) et l'Analyse en Composantes Principales (PCA).

Fonctionnalités:
- Détection de vulnérabilités structurelles
- Modélisation de perturbations stochastiques
- Analyse de robustesse réseau
- Reconstruction par réduction de dimension

Auteur: Refactored by Antigravity
Date: 2026-05-30
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.collections import LineCollection
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd, eig
from scipy.sparse import csr_matrix
import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings
import os

warnings.filterwarnings('ignore')

# Configuration des styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# STRUCTURES DE DONNÉES
# =============================================================================

@dataclass
class NetworkVulnerability:
    """Structure pour stocker les vulnérabilités détectées"""
    node_id: int
    vulnerability_score: float
    centrality_rank: int
    perturbation_sensitivity: float
    attack_vector: str


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================

class NetworkSVDAnalyzer:
    """
    Moteur d'analyse de réseau basé sur SVD et PCA
    pour la détection de vulnérabilités et la modélisation
    des perturbations stochastiques.
    """
    
    def __init__(self, n_nodes: int = 50, connection_prob: float = 0.15, 
                 seed: int = 42, attack_budget: float = 0.3):
        self.n_nodes = n_nodes
        self.connection_prob = connection_prob
        self.seed = seed
        self.attack_budget = attack_budget
        np.random.seed(seed)
        
        # Génération du réseau
        self.G = self._generate_critical_network()
        self.adj_matrix = nx.to_numpy_array(self.G)
        self.laplacian = nx.laplacian_matrix(self.G).toarray()
        
        # Matrices de covariance et corrélation
        self.cov_matrix = np.cov(self.adj_matrix)
        self.corr_matrix = np.corrcoef(self.adj_matrix)
        
        # Résultats SVD
        self.U = None
        self.S = None  # Valeurs singulières
        self.Vt = None
        self.pca = None
        self.explained_variance_ratio = None
        
        # Analyse des vulnérabilités
        self.vulnerabilities: List[NetworkVulnerability] = []
        self.perturbation_vectors = []
        
    def _generate_critical_network(self):
        """Génère un réseau critique avec des nœuds de différentes importances"""
        G = nx.Graph()
        G.add_nodes_from(range(self.n_nodes))
        
        # Nœuds critiques (hubs) - cibles prioritaires
        critical_nodes = [0, 1, 2, 3, 4]
        
        # Connexions préférentielles vers les hubs
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                prob = self.connection_prob
                # Augmenter la probabilité de connexion aux nœuds critiques
                if i in critical_nodes or j in critical_nodes:
                    prob *= 3.0
                if np.random.random() < prob:
                    weight = np.random.uniform(0.5, 5.0)
                    if i in critical_nodes and j in critical_nodes:
                        weight *= 2.0  # Connexions inter-hubs plus fortes
                    G.add_edge(i, j, weight=weight)
        
        # S'assurer que le graphe est connexe
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                node_a = list(components[i])[0]
                node_b = list(components[i + 1])[0]
                G.add_edge(node_a, node_b, weight=1.0)
        
        return G
    
    def perform_svd_decomposition(self):
        """Étape 1 & 2 : Décomposition SVD de la matrice d'adjacence"""
        print("\n" + "=" * 60)
        print("PHASE 1 : DÉCOMPOSITION SVD")
        print("=" * 60)
        
        # SVD complète
        self.U, self.S, self.Vt = svd(self.adj_matrix, full_matrices=False)
        
        print(f"Dimensions de la matrice d'adjacence : {self.adj_matrix.shape}")
        print(f"Rang effectif du réseau : {np.sum(self.S > 1e-10)}")
        print(f"\nTop 10 Valeurs Singulières :")
        for i, s in enumerate(self.S[:10]):
            print(f"  s_{i+1:2d} = {s:.4f}  (Variance expliquée: {s**2/np.sum(self.S**2)*100:.2f}%)")
        
        # PCA pour analyse de variance
        scaler = StandardScaler()
        adj_scaled = scaler.fit_transform(self.adj_matrix)
        self.pca = PCA()
        self.pca.fit(adj_scaled)
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        
        return self.S, self.explained_variance_ratio
    
    def analyze_vulnerabilities(self):
        """Identification des vulnérabilités via analyse spectrale"""
        print("\n" + "=" * 60)
        print("PHASE 2 : ANALYSE DES VULNÉRABILITÉS")
        print("=" * 60)
        
        # 1. Centralité basée sur les vecteurs propres (Eigenvector Centrality)
        eigen_centrality = nx.eigenvector_centrality(self.G, weight='weight', max_iter=1000)
        
        # 2. Centralité d'intermédiarité (Betweenness)
        betweenness = nx.betweenness_centrality(self.G, weight='weight')
        
        # 3. Centralité de proximité
        closeness = nx.closeness_centrality(self.G)
        
        # 4. Analyse de la sensibilité aux perturbations (via SVD)
        if self.U is None:
            self.perform_svd_decomposition()
            
        sensitivity = np.abs(self.U[:, 0]) + np.abs(self.U[:, 1])
        sensitivity = sensitivity / np.max(sensitivity)
        
        # Score composite de vulnérabilité
        self.vulnerabilities = []
        for node in range(self.n_nodes):
            vuln_score = (
                0.35 * eigen_centrality[node] +
                0.30 * betweenness[node] +
                0.15 * closeness[node] +
                0.20 * sensitivity[node]
            )
            
            # Classification du vecteur d'attaque
            if eigen_centrality[node] > 0.8:
                attack_vec = "Attaque par déni de service distribué (DDoS)"
            elif betweenness[node] > 0.6:
                attack_vec = "Interception de flux de données (Man-in-the-Middle)"
            elif sensitivity[node] > 0.7:
                attack_vec = "Injection de perturbations stochastiques"
            else:
                attack_vec = "Exploitation de vulnérabilité locale"
            
            self.vulnerabilities.append(NetworkVulnerability(
                node_id=node,
                vulnerability_score=vuln_score,
                centrality_rank=0,  # Sera mis à jour
                perturbation_sensitivity=sensitivity[node],
                attack_vector=attack_vec
            ))
        
        # Tri par score de vulnérabilité
        self.vulnerabilities.sort(key=lambda x: x.vulnerability_score, reverse=True)
        for i, vuln in enumerate(self.vulnerabilities):
            vuln.centrality_rank = i + 1
        
        # Affichage des top vulnérabilités
        print(f"\nTop 10 Nœuds Critiques (Vulnérabilités) :")
        print(f"{'Rank':<6} {'Node':<6} {'Score':<10} {'Sensibilité':<12} {'Vecteur d\'attaque'}")
        print("-" * 80)
        for vuln in self.vulnerabilities[:10]:
            print(f"{vuln.centrality_rank:<6} {vuln.node_id:<6} {vuln.vulnerability_score:.4f}    "
                  f"{vuln.perturbation_sensitivity:.4f}       {vuln.attack_vector}")
        
        return self.vulnerabilities
    
    def model_stochastic_perturbations(self, n_perturbations: int = 100):
        """Modélisation des vecteurs de perturbation stochastique locale"""
        print("\n" + "=" * 60)
        print("PHASE 3 : MODÉLISATION DES PERTURBATIONS STOCHASTIQUES")
        print("=" * 60)
        
        if self.U is None:
            self.perform_svd_decomposition()
            
        # Génération de perturbations dans l'espace réduit (SVD)
        k = min(10, self.n_nodes // 2)  # Dimension réduite
        
        perturbations = []
        impacts = []
        
        for i in range(n_perturbations):
            # Vecteur de perturbation dans l'espace des valeurs singulières
            delta_s = np.random.normal(0, self.attack_budget, k)
            
            # Projection dans l'espace original
            perturbation = self.U[:, :k] @ np.diag(self.S[:k]) @ delta_s
            perturbation = perturbation / np.linalg.norm(perturbation)
            
            # Calcul de l'impact sur la connectivité
            perturbed_adj = self.adj_matrix + np.outer(perturbation, perturbation) * 0.1
            
            # Mesure de l'impact : changement dans les valeurs singulières
            S_perturbed = svd(perturbed_adj, compute_uv=False)
            impact = np.sum(np.abs(self.S - S_perturbed)[:k])
            
            perturbations.append(perturbation)
            impacts.append(impact)
        
        self.perturbation_vectors = perturbations
        
        print(f"Nombre de perturbations simulées : {n_perturbations}")
        print(f"Dimension de réduction (k) : {k}")
        print(f"Impact moyen des perturbations : {np.mean(impacts):.4f}")
        print(f"Impact maximal détecté : {np.max(impacts):.4f}")
        print(f"\nAnalyse : Les perturbations ciblant les {k} premières")
        print(f"composantes singulières ont le plus grand effet sur la structure du réseau.")
        
        return perturbations, impacts
    
    def reconstruct_reduced_network(self, k: int = 5):
        """Étape 3 : Reconstruction du réseau avec k composantes principales"""
        print("\n" + "=" * 60)
        print(f"PHASE 4 : RECONSTRUCTION RÉDUITE (k={k})")
        print("=" * 60)
        
        if self.U is None:
            self.perform_svd_decomposition()
            
        # Reconstruction SVD tronquée
        S_k = np.zeros_like(self.S)
        S_k[:k] = self.S[:k]
        
        adj_reconstructed = self.U @ np.diag(S_k) @ self.Vt
        
        # Erreur de reconstruction
        reconstruction_error = np.linalg.norm(self.adj_matrix - adj_reconstructed, 'fro')
        relative_error = reconstruction_error / np.linalg.norm(self.adj_matrix, 'fro')
        
        # Variance expliquée
        variance_explained = np.sum(self.S[:k]**2) / np.sum(self.S**2)
        
        print(f"Erreur de reconstruction (Frobenius) : {reconstruction_error:.4f}")
        print(f"Erreur relative : {relative_error*100:.2f}%")
        print(f"Variance expliquée par {k} composantes : {variance_explained*100:.2f}%")
        
        # Création du graphe reconstruit
        threshold = np.mean(adj_reconstructed)
        adj_binary = (adj_reconstructed > threshold).astype(int)
        np.fill_diagonal(adj_binary, 0)
        
        G_reconstructed = nx.from_numpy_array(adj_binary)
        
        print(f"\nComparaison des propriétés structurelles :")
        print(f"  Nœuds originaux : {self.G.number_of_nodes()}")
        print(f"  Arêtes originales : {self.G.number_of_edges()}")
        print(f"  Arêtes reconstruites : {G_reconstructed.number_of_edges()}")
        
        return adj_reconstructed, G_reconstructed, variance_explained


# =============================================================================
# FONCTIONS DE VISUALISATION
# =============================================================================

def create_dashboard(analyzer, perturbations, impacts, adj_reconstructed, G_reconstructed, variance, output_path='network_svd_dashboard.png'):
    """Crée le dashboard principal de visualisation"""
    
    # Assurer que le dossier parent existe
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle('ANALYSE DE RÉSEAU CRITIQUE - SVD & PCA\nDétection de Vulnérabilités et Perturbations Stochastiques', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Panel 1: Réseau Original
    ax1 = fig.add_subplot(2, 3, 1)
    pos = nx.spring_layout(analyzer.G, k=2, iterations=50, seed=42)
    vuln_scores = [v.vulnerability_score for v in analyzer.vulnerabilities]
    node_colors = []
    node_sizes = []
    for node in analyzer.G.nodes():
        score = next((v.vulnerability_score for v in analyzer.vulnerabilities if v.node_id == node), 0)
        node_colors.append(score)
        node_sizes.append(300 + score * 1000)
    
    edges = analyzer.G.edges()
    weights = [analyzer.G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(analyzer.G, pos, alpha=0.3, width=[w/3 for w in weights], ax=ax1, edge_color='gray')
    nodes = nx.draw_networkx_nodes(analyzer.G, pos, node_color=node_colors, node_size=node_sizes, 
                                    cmap=plt.cm.Reds, ax=ax1, vmin=0, vmax=max(vuln_scores))
    critical_labels = {v.node_id: f"N{v.node_id}\n({v.vulnerability_score:.2f})" 
                       for v in analyzer.vulnerabilities[:8]}
    nx.draw_networkx_labels(analyzer.G, pos, critical_labels, font_size=8, font_weight='bold', ax=ax1)
    ax1.set_title('Réseau Critique\n(Couleur = Vulnérabilité)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=max(vuln_scores)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, shrink=0.8)
    
    # Panel 2: Valeurs Singulières
    ax2 = fig.add_subplot(2, 3, 2)
    x = np.arange(1, len(analyzer.S) + 1)
    ax2.bar(x[:20], analyzer.S[:20], color='steelblue', alpha=0.7, label='Valeurs Singulières')
    ax2.set_xlabel('Rang (k)', fontsize=12)
    ax2.set_ylabel('Valeur Singulière (σ)', fontsize=12, color='steelblue')
    ax2_twin = ax2.twinx()
    cumvar = np.cumsum(analyzer.S**2) / np.sum(analyzer.S**2) * 100
    ax2_twin.plot(x[:20], cumvar[:20], 'ro-', linewidth=2, markersize=6, label='Variance Cumulative')
    ax2_twin.axhline(y=85, color='green', linestyle='--', alpha=0.7, label='Seuil 85%')
    ax2_twin.set_ylabel('Variance Expliquée Cumulative (%)', fontsize=12, color='red')
    ax2.set_title('Spectre des Valeurs Singulières\net Variance Expliquée', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='center right')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Matrice d'Adjacence
    ax3 = fig.add_subplot(2, 3, 3)
    im1 = ax3.imshow(analyzer.adj_matrix[:30, :30], cmap='viridis', aspect='auto')
    ax3.set_title('Matrice d\'Adjacence (30x30)', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax3, shrink=0.8)
    
    # Panel 4: Perturbations Stochastiques
    ax4 = fig.add_subplot(2, 3, 4)
    top_nodes = [v.node_id for v in analyzer.vulnerabilities[:15]]
    perturbation_matrix = np.array(perturbations)
    pert_subset = perturbation_matrix[:, top_nodes]
    im2 = ax4.imshow(pert_subset[:50].T, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    ax4.set_title('Vecteurs de Perturbation Stochastique\n(Nœuds Critiques)', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax4, shrink=0.8)
    
    # Panel 5: Projection PCA
    ax5 = fig.add_subplot(2, 3, 5)
    pca_2d = PCA(n_components=2)
    adj_pca = pca_2d.fit_transform(StandardScaler().fit_transform(analyzer.adj_matrix))
    scatter = ax5.scatter(adj_pca[:, 0], adj_pca[:, 1], 
                          c=[v.vulnerability_score for v in analyzer.vulnerabilities],
                          s=[v.perturbation_sensitivity * 500 for v in analyzer.vulnerabilities],
                          cmap='YlOrRd', alpha=0.8, edgecolors='black', linewidths=0.5)
    for i, vuln in enumerate(analyzer.vulnerabilities[:5]):
        ax5.annotate(f'N{vuln.node_id}', 
                    (adj_pca[vuln.node_id, 0], adj_pca[vuln.node_id, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    ax5.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax5.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax5.set_title('Projection PCA du Réseau\n(Espace Réduit)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax5, shrink=0.8)
    
    # Panel 6: Erreur de Reconstruction
    ax6 = fig.add_subplot(2, 3, 6)
    k_show = 10
    S_k = np.zeros_like(analyzer.S)
    S_k[:k_show] = analyzer.S[:k_show]
    adj_recon = analyzer.U @ np.diag(S_k) @ analyzer.Vt
    error_matrix = np.abs(analyzer.adj_matrix - adj_recon)
    im3 = ax6.imshow(error_matrix[:30, :30], cmap='hot', aspect='auto')
    ax6.set_title(f'Erreur de Reconstruction\n(k={k_show} composantes)', fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=ax6, shrink=0.8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Dashboard sauvegardé: {output_path}")


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("APPLICATION D'ANALYSE DE RÉSEAU - SVD & PCA")
    print("Réduction de dimension et détection de vulnérabilités")
    print("=" * 80)
    
    # Initialisation
    analyzer = NetworkSVDAnalyzer(n_nodes=50, connection_prob=0.12, attack_budget=0.25)
    
    # Phase 1: SVD
    singular_values, variance_ratio = analyzer.perform_svd_decomposition()
    
    # Phase 2: Vulnérabilités
    vulnerabilities = analyzer.analyze_vulnerabilities()
    
    # Phase 3: Perturbations stochastiques
    perturbations, impacts = analyzer.model_stochastic_perturbations(n_perturbations=200)
    
    # Phase 4: Reconstruction
    k_optimal = np.argmax(np.cumsum(variance_ratio) > 0.85) + 1
    adj_reconstructed, G_reconstructed, variance = analyzer.reconstruct_reduced_network(k=k_optimal)
    
    # Visualisation
    create_dashboard(analyzer, perturbations, impacts, adj_reconstructed, G_reconstructed, variance)
    
    print("\n" + "=" * 60)
    print("ANALYSE COMPLÉTÉE")
    print("=" * 60)
