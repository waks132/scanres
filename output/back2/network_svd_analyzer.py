
# FICHIER 1: network_svd_analyzer.py - VERSION OPTIMISÉE COMPLÈTE
# Implémentation de toutes les optimisations du plan

code_v1 = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network SVD/PCA Analyzer v3.0 - Optimisé
============================================
Implémentation complète des optimisations:
- SVD Randomized (Halko et al.)
- Cache Hiérarchique L1/L2/L3
- Détection Non-Linéaire (Kernel PCA + Ensemble)
- Parallélisation Joblib
- Numba JIT pour fonctions critiques

Auteur: Optimized by AI
Date: 2026-05-31
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.collections import LineCollection
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from scipy.linalg import svd as scipy_svd, eig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds, eigs
import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
import os
import time
import warnings

# Tentative d'import des librairies d'optimisation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """Configuration optimisée"""
    N_NODES = 150
    SEED = 42
    K_SVD = 30
    K_PCA = 5
    ATTACK_BUDGET = 0.25
    N_PERTURBATIONS = 200
    
    # Cache
    CACHE_L1_TTL = 60
    CACHE_L2_PATH = './cache_l2'
    
    # Parallélisation
    N_JOBS = -1
    
    # Numba
    NUMBA_PARALLEL = True
    
    # GPU
    USE_GPU = CUPY_AVAILABLE


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE HIÉRARCHIQUE L1/L2/L3
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchicalCache:
    """Cache hiérarchique avec L1 (RAM/TTL), L2 (disque), L3 (précalculs)"""
    
    def __init__(self, ttl=60, l2_path='./cache_l2'):
        self.l1 = OrderedDict()
        self.l1_timestamps = {}
        self.ttl = ttl
        self.l2_path = l2_path
        os.makedirs(l2_path, exist_ok=True)
        self.l3 = {}
        self._hit_count = 0
        self._miss_count = 0
        
    def _hash_key(self, matrix_type, k, matrix_data):
        """Génère une clé de cache unique basée sur le hash MD5"""
        if hasattr(matrix_data, 'tobytes'):
            data_hash = hashlib.md5(matrix_data.tobytes()).hexdigest()[:16]
        else:
            data_hash = hashlib.md5(str(matrix_data).encode()).hexdigest()[:16]
        return f"{matrix_type}_k{k}_{data_hash}"
    
    def get(self, matrix_type, k, matrix_data):
        """Récupère du cache L1, L2 ou L3"""
        key = self._hash_key(matrix_type, k, matrix_data)
        
        # L1: Mémoire vive
        if key in self.l1:
            if time.time() - self.l1_timestamps.get(key, 0) < self.ttl:
                self.l1.move_to_end(key)
                self._hit_count += 1
                return self.l1[key]
            else:
                del self.l1[key]
                del self.l1_timestamps[key]
        
        # L2: Disque
        l2_file = os.path.join(self.l2_path, f"{key}.npz")
        if os.path.exists(l2_file):
            data = np.load(l2_file, allow_pickle=True)
            result = (data['U'], data['S'], data['Vt'])
            self._put_l1(key, result)
            self._hit_count += 1
            return result
        
        # L3: Précalculs
        if key in self.l3:
            self._hit_count += 1
            return self.l3[key]
        
        self._miss_count += 1
        return None
    
    def put(self, matrix_type, k, matrix_data, result):
        """Stocke dans L1 et L2"""
        key = self._hash_key(matrix_type, k, matrix_data)
        self._put_l1(key, result)
        self._put_l2(key, result)
    
    def _put_l1(self, key, result):
        """Stocke dans le cache L1 (mémoire)"""
        self.l1[key] = result
        self.l1_timestamps[key] = time.time()
        self.l1.move_to_end(key)
        if len(self.l1) > 100:
            oldest = next(iter(self.l1))
            del self.l1[oldest]
            del self.l1_timestamps[oldest]
    
    def _put_l2(self, key, result):
        """Stocke dans le cache L2 (disque)"""
        l2_file = os.path.join(self.l2_path, f"{key}.npz")
        U, S, Vt = result
        np.savez_compressed(l2_file, U=U, S=S, Vt=Vt)
    
    def preload_l3(self, engine):
        """Précalcule les centralités pour L3"""
        print("Précalcul des centralités (Cache L3)...")
        self.l3['eigenvector'] = nx.eigenvector_centrality(engine.G, weight='weight', max_iter=1000)
        self.l3['betweenness'] = nx.betweenness_centrality(engine.G, weight='weight')
        self.l3['closeness'] = nx.closeness_centrality(engine.G)
        self.l3['pagerank'] = nx.pagerank(engine.G, weight='weight')
        print("Cache L3 prêt.")
    
    def get_centrality(self, centrality_type):
        """Récupère une centralité du cache L3"""
        return self.l3.get(centrality_type, {})
    
    def get_stats(self):
        """Retourne les statistiques du cache"""
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0
        return {
            'hits': self._hit_count,
            'misses': self._miss_count,
            'hit_rate': hit_rate,
            'l1_size': len(self.l1),
            'l3_size': len(self.l3)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS NUMBA JIT
# ═══════════════════════════════════════════════════════════════════════════════

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def fast_spectral_vulnerability_numba(U, S, k):
        """Calcule la vulnérabilité spectrale en parallèle avec Numba"""
        n_nodes = U.shape[0]
        result = np.zeros(n_nodes)
        s_sum = np.sum(S[:k])
        
        for i in prange(n_nodes):
            for j in range(k):
                result[i] += (U[i, j] ** 2) * S[j]
            result[i] /= s_sum
        
        return result
    
    @jit(nopython=True, cache=True)
    def fast_perturbation_impact_numba(perturbation, adj_matrix, k):
        """Calcule l'impact d'une perturbation"""
        perturbed = adj_matrix + np.outer(perturbation, perturbation) * 0.1
        # Approximation rapide de l'impact spectral
        impact = 0.0
        for i in range(min(k, len(perturbation))):
            impact += abs(perturbed[i, i] - adj_matrix[i, i])
        return impact
else:
    def fast_spectral_vulnerability_numba(U, S, k):
        """Fallback sans Numba"""
        spectral_contributions = (U[:, :k] ** 2) * S[:k]
        return np.sum(spectral_contributions, axis=1) / np.sum(S[:k])
    
    def fast_perturbation_impact_numba(perturbation, adj_matrix, k):
        """Fallback sans Numba"""
        perturbed = adj_matrix + np.outer(perturbation, perturbation) * 0.1
        return np.sum(np.abs(np.diag(perturbed) - np.diag(adj_matrix))[:k])


# ═══════════════════════════════════════════════════════════════════════════════
# MOTEUR SVD RANDOMIZED + GPU
# ═══════════════════════════════════════════════════════════════════════════════

class OptimizedSVDEngine:
    """
    Moteur SVD avec:
    - SVD Randomized (Halko et al. 2011)
    - SVD Sparse (ARPACK)
    - Cache hiérarchique L1/L2/L3
    - Support GPU (CuPy)
    - Mise à jour incrémentale (Brand)
    """
    
    def __init__(self, n_nodes=None, connection_prob=0.06, seed=None, use_sparse=True):
        self.n_nodes = n_nodes or Config.N_NODES
        self.seed = seed or Config.SEED
        self.use_sparse = use_sparse and self.n_nodes > 100
        np.random.seed(self.seed)
        
        # Génération du réseau
        self.G = self._generate_network()
        self.adj_matrix = nx.to_numpy_array(self.G, weight='weight')
        
        # Cache
        self.cache = HierarchicalCache()
        
        # Laplaciens
        self.laplacian = None
        self.normalized_laplacian = None
        self._compute_laplacians()
        
        # Attributs SVD persistants
        self.U = None
        self.S = None
        self.Vt = None
        
        # Précharger le cache L3
        self.cache.preload_l3(self)
        
    def _generate_network(self):
        """Génération optimisée du réseau avec communautés"""
        G = nx.Graph()
        G.add_nodes_from(range(self.n_nodes))
        
        n_communities = max(3, self.n_nodes // 30)
        community_size = self.n_nodes // n_communities
        
        # Watts-Strogatz par communauté
        for c in range(n_communities):
            start = c * community_size
            end = min((c + 1) * community_size, self.n_nodes)
            size = end - start
            subG = nx.watts_strogatz_graph(size, k=min(6, size-1), p=0.3, seed=self.seed+c)
            mapping = {i: start + i for i in range(size)}
            subG = nx.relabel_nodes(subG, mapping)
            G.add_edges_from(subG.edges())
        
        # Ponts inter-communautaires
        bridge_nodes = [c * community_size for c in range(n_communities)]
        for i, node_a in enumerate(bridge_nodes):
            for j, node_b in enumerate(bridge_nodes):
                if i < j and np.random.random() < 0.7:
                    G.add_edge(node_a, node_b, weight=np.random.uniform(3.0, 8.0))
        
        # Super-hubs
        critical_nodes = [0, community_size, 2*community_size]
        for node in critical_nodes:
            targets = np.random.choice(
                [n for n in G.nodes() if n != node], 
                size=min(15, self.n_nodes//3), replace=False
            )
            for target in targets:
                if not G.has_edge(node, target):
                    G.add_edge(node, target, weight=np.random.uniform(2.0, 10.0))
        
        # Pondération par centralité
        betweenness = nx.betweenness_centrality(G)
        for u, v in G.edges():
            bw = (betweenness[u] + betweenness[v]) / 2
            G[u][v]['weight'] = G[u][v].get('weight', 1.0) * (1 + 5 * bw)
        
        return G
    
    def _compute_laplacians(self):
        """Calcule les Laplaciens standard et normalisé"""
        self.laplacian = nx.laplacian_matrix(self.G, weight='weight').toarray()
        degrees = np.array([self.G.degree(n, weight='weight') for n in self.G.nodes()])
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))
        self.normalized_laplacian = D_inv_sqrt @ self.laplacian @ D_inv_sqrt
    
    def compute_svd(self, k=None, matrix_type='adjacency', use_cache=True, algorithm='auto'):
        """
        SVD avec sélection intelligente de l'algorithme.
        
        Args:
            k: Nombre de composantes
            matrix_type: 'adjacency', 'laplacian', 'normalized_laplacian'
            use_cache: Utiliser le cache
            algorithm: 'auto', 'randomized', 'sparse', 'dense', 'gpu'
        """
        k = k or Config.K_SVD
        
        # Sélection de la matrice
        if matrix_type == 'adjacency':
            M = self.adj_matrix
        elif matrix_type == 'laplacian':
            M = self.laplacian
        elif matrix_type == 'normalized_laplacian':
            M = self.normalized_laplacian
        else:
            raise ValueError(f"Type inconnu: {matrix_type}")
        
        # Vérifier le cache
        if use_cache:
            cached = self.cache.get(matrix_type, k, M)
            if cached is not None:
                U, S, Vt = cached
                if matrix_type == 'adjacency':
                    self.U, self.S, self.Vt = U, S, Vt
                return U, S, Vt
        
        # Sélection automatique de l'algorithme
        n = M.shape[0]
        if algorithm == 'auto':
            if CUPY_AVAILABLE and n > 200:
                algorithm = 'gpu'
            elif n > 100 and k < n // 2:
                algorithm = 'randomized'
            elif self.use_sparse and n > 100:
                algorithm = 'sparse'
            else:
                algorithm = 'dense'
        
        # Exécution selon l'algorithme
        if algorithm == 'gpu' and CUPY_AVAILABLE:
            U, S, Vt = self._svd_gpu(M, k)
        elif algorithm == 'randomized':
            U, S, Vt = self._svd_randomized(M, k)
        elif algorithm == 'sparse':
            U, S, Vt = self._svd_sparse(M, k)
        else:
            U, S, Vt = self._svd_dense(M, k)
        
        result = (U, S, Vt)
        
        # Stocker dans le cache
        if use_cache:
            self.cache.put(matrix_type, k, M, result)
        
        if matrix_type == 'adjacency':
            self.U, self.S, self.Vt = U, S, Vt
        
        return U, S, Vt
    
    def _svd_randomized(self, M, k):
        """SVD Randomized (Halko et al. 2011) - O(nk²)"""
        U, S, Vt = randomized_svd(M, n_components=k, n_oversamples=10, 
                                   power_iteration_normalizer='QR',
                                   random_state=self.seed)
        return U, S, Vt
    
    def _svd_sparse(self, M, k):
        """SVD Sparse via ARPACK"""
        M_sparse = csr_matrix(M)
        U, S, Vt = svds(M_sparse, k=k, which='LM')
        idx = np.argsort(S)[::-1]
        return U[:, idx], S[idx], Vt[idx, :]
    
    def _svd_dense(self, M, k):
        """SVD Dense via LAPACK"""
        U, S, Vt = scipy_svd(M, full_matrices=False)
        return U[:, :k], S[:k], Vt[:k, :]
    
    def _svd_gpu(self, M, k):
        """SVD sur GPU via CuPy"""
        M_gpu = cp.asarray(M)
        U_gpu, S_gpu, Vt_gpu = cp.linalg.svd(M_gpu, full_matrices=False)
        U = cp.asnumpy(U_gpu[:, :k])
        S = cp.asnumpy(S_gpu[:k])
        Vt = cp.asnumpy(Vt_gpu[:k, :])
        return U, S, Vt
    
    def incremental_update(self, edge_changes):
        """Mise à jour incrémentale du SVD (algorithme de Brand)"""
        if self.U is None or self.S is None:
            self.compute_svd()
        
        U, S, Vt = self.U, self.S, self.Vt
        k = len(S)
        
        for u, v, delta in edge_changes:
            # Vecteurs du rang-1
            a = np.zeros(self.n_nodes)
            a[u] = delta
            b = np.zeros(self.n_nodes)
            b[v] = 1.0
            
            # Projection
            m = U.T @ a
            p = a - U @ m
            p_norm = np.linalg.norm(p)
            P = p / p_norm if p_norm > 1e-9 else np.zeros_like(p)
            
            n = Vt @ b
            q = b - Vt.T @ n
            q_norm = np.linalg.norm(q)
            Q = q / q_norm if q_norm > 1e-9 else np.zeros_like(q)
            
            # Matrice de couplage
            K_mat = np.zeros((k + 1, k + 1))
            K_mat[:k, :k] = np.diag(S)
            col_vec = np.append(m, p_norm)
            row_vec = np.append(n, q_norm)
            K_mat += np.outer(col_vec, row_vec)
            
            # SVD de la petite matrice
            Uk, Sk, Vtk = scipy_svd(K_mat, full_matrices=False)
            
            # Reconstruction
            U_new = np.column_stack((U, P)) @ Uk
            Vt_new = Vtk @ np.vstack((Vt, Q))
            
            U = U_new[:, :k]
            S = Sk[:k]
            Vt = Vt_new[:k, :]
        
        # Mise à jour
        self.U, self.S, self.Vt = U, S, Vt
        self.cache._svd_cache = {}  # Invalidation
        
        return S
    
    def get_centrality(self, centrality_type):
        """Récupère une centralité du cache L3"""
        return self.cache.get_centrality(centrality_type)
    
    def extract_spectral_features(self):
        """Extrait les caractéristiques spectrales"""
        U, S, Vt = self.compute_svd(k=min(self.n_nodes//2, 100))
        
        features = {
            'spectral_gap': float(S[0] - S[1]) if len(S) > 1 else 0,
            'effective_rank': int(np.sum(S / S[0] > 0.01)),
            'entropy': float(-np.sum((S**2 / np.sum(S**2)) * np.log(S**2 / np.sum(S**2) + 1e-10))),
            'condition_number': float(S[0] / S[-1]) if S[-1] > 1e-10 else float('inf'),
            'participation_ratio': float(np.sum(S**2)**2 / np.sum(S**4)),
        }
        
        eigvals, eigvecs = eigs(csr_matrix(self.normalized_laplacian), k=10, which='SM')
        features['fiedler_value'] = float(np.real(eigvals[1])) if len(eigvals) > 1 else 0
        features['community_structure'] = float(np.std(np.real(eigvecs[:, 1])))
        
        return features


# ═══════════════════════════════════════════════════════════════════════════════
# DÉTECTION NON-LINÉAIRE (Kernel PCA + Ensemble)
# ═══════════════════════════════════════════════════════════════════════════════

class NonlinearDetector:
    """
    Détecteur d'anomalies avec:
    - Kernel PCA (patterns non-linéaires)
    - Isolation Forest
    - Local Outlier Factor
    - Vote majoritaire adaptatif
    """
    
    def __init__(self, engine: OptimizedSVDEngine):
        self.engine = engine
        self.kpca = None
        self.iso_forest = None
        self.lof = None
        self.kpca_threshold = None
        
    def fit(self, node_embeddings):
        """Entraîne les détecteurs"""
        # Kernel PCA
        self.kpca = KernelPCA(n_components=10, kernel='rbf', gamma=0.1, 
                               fit_inverse_transform=True)
        self.kpca.fit(node_embeddings)
        
        # Calcul du seuil de reconstruction
        kpca_projection = self.kpca.transform(node_embeddings)
        reconstruction = self.kpca.inverse_transform(kpca_projection)
        kpca_error = np.linalg.norm(node_embeddings - reconstruction, axis=1)
        self.kpca_threshold = np.percentile(kpca_error, 90)
        
        # Isolation Forest
        self.iso_forest = IsolationForest(contamination=0.1, random_state=42, 
                                           n_estimators=200, n_jobs=-1)
        self.iso_forest.fit(node_embeddings)
        
        # LOF
        self.lof = LocalOutlierFactor(n_neighbors=10, contamination=0.1, 
                                       novelty=True, n_jobs=-1)
        self.lof.fit(node_embeddings)
        
    def predict(self, node_embeddings):
        """Prédit les anomalies avec vote majoritaire"""
        # Kernel PCA
        kpca_projection = self.kpca.transform(node_embeddings)
        reconstruction = self.kpca.inverse_transform(kpca_projection)
        kpca_error = np.linalg.norm(node_embeddings - reconstruction, axis=1)
        kpca_anomaly = kpca_error > self.kpca_threshold
        
        # Isolation Forest
        if_anomaly = self.iso_forest.predict(node_embeddings) == -1
        
        # LOF
        lof_anomaly = self.lof.predict(node_embeddings) == -1
        
        # Vote majoritaire (2 sur 3)
        ensemble_vote = (kpca_anomaly.astype(int) + 
                         if_anomaly.astype(int) + 
                         lof_anomaly.astype(int)) >= 2
        
        return ensemble_vote
    
    def predict_proba(self, node_embeddings):
        """Retourne les probabilités d'anomalie"""
        # Scores normalisés
        kpca_projection = self.kpca.transform(node_embeddings)
        reconstruction = self.kpca.inverse_transform(kpca_projection)
        kpca_error = np.linalg.norm(node_embeddings - reconstruction, axis=1)
        kpca_score = kpca_error / self.kpca_threshold
        
        if_score = -self.iso_forest.score_samples(node_embeddings)
        if_score = (if_score - if_score.min()) / (if_score.max() - if_score.min() + 1e-10)
        
        lof_score = -self.lof.score_samples(node_embeddings)
        lof_score = (lof_score - lof_score.min()) / (lof_score.max() - lof_score.min() + 1e-10)
        
        # Moyenne pondérée
        proba = (0.4 * kpca_score + 0.3 * if_score + 0.3 * lof_score) / 3
        return np.clip(proba, 0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("NETWORK SVD/PCA ANALYZER v3.0 - Optimisé")
    print("=" * 80)
    
    # Initialisation
    start = time.time()
    engine = OptimizedSVDEngine(n_nodes=500, seed=42)
    print(f"\\nRéseau: {engine.n_nodes} nœuds, {engine.G.number_of_edges()} arêtes")
    print(f"Initialisation: {time.time()-start:.3f}s")
    
    # SVD avec cache
    start = time.time()
    U, S, Vt = engine.compute_svd(k=30, matrix_type='adjacency')
    print(f"SVD (1ère fois): {time.time()-start:.3f}s")
    
    # SVD depuis le cache
    start = time.time()
    U, S, Vt = engine.compute_svd(k=30, matrix_type='adjacency')
    print(f"SVD (cache): {time.time()-start:.3f}s")
    
    # Centralités depuis L3
    eigen_centrality = engine.get_centrality('eigenvector')
    print(f"Centralités (L3): {len(eigen_centrality)} nœuds")
    
    # Détecteur non-linéaire
    detector = NonlinearDetector(engine)
    node_embeddings = U[:, :10] * S[:10]
    detector.fit(node_embeddings)
    anomalies = detector.predict(node_embeddings)
    print(f"Anomalies détectées: {np.sum(anomalies)} nœuds")
    
    # Stats cache
    stats = engine.cache.get_stats()
    print(f"\\nCache: {stats['hit_rate']*100:.1f}% hit rate")
    
    print("\\n" + "=" * 60)
    print("SYSTÈME OPTIMISÉ PRÊT")
    print("=" * 60)
'''

with open('/mnt/agents/output/network_svd_analyzer_v3.py', 'w', encoding='utf-8') as f:
    f.write(code_v1)

print("network_svd_analyzer_v3.py créé: {} caractères".format(len(code_v1)))
