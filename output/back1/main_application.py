
# FICHIER 3: main_application_v3.py - INTERFACE UNIFIÉE COMPLÈTE

code_v3 = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Security Analyzer v3.0 - Application Complète Optimisée
=================================================================
Application unifiée d'analyse de réseau par SVD/PCA avec toutes les optimisations:

Optimisations implémentées:
- SVD Randomized (Halko et al. 2011) - O(nk²) vs O(n³)
- Cache Hiérarchique L1/L2/L3 (RAM, Disque, Précalculs)
- Détection Non-Linéaire (Kernel PCA + Isolation Forest + LOF)
- Parallélisation Joblib pour scan de vulnérabilités
- Évasion par Apprentissage par Renforcement (PPO)
- IDS Dynamique avec Online Learning (River) et ADWIN
- Transfer Learning avancé (Procrustes + GAN)
- Support GPU (CuPy)
- Numba JIT pour fonctions critiques

Modules:
    network_svd_analyzer_v3.py      - Moteur SVD/PCA optimisé
    optimized_offensive_engine_v3.py  - Intelligence offensive proactive
    main_application_v3.py          - Interface unifiée (ce fichier)

Usage:
    python main_application_v3.py --mode basic      # Analyse de base
    python main_application_v3.py --mode offensive  # Mode offensif
    python main_application_v3.py --mode full       # Mode complet
    python main_application_v3.py --nodes 1000 --gpu  # Grande échelle + GPU

Dépendances:
    numpy, scipy, matplotlib, seaborn, scikit-learn, networkx
    Optionnel: numba, joblib, cupy, torch, river

Auteur: Optimized by AI
Date: 2026-05-31
"""

import argparse
import sys
import os
import numpy as np
import time

# Ajouter le répertoire courant au path pour les imports locaux
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports des modules métiers optimisés
try:
    from network_svd_analyzer_v3 import (
        OptimizedSVDEngine, 
        NonlinearDetector,
        HierarchicalCache,
        fast_spectral_vulnerability_numba,
        Config as SVDConfig
    )
    SVD_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: network_svd_analyzer_v3 non disponible: {e}")
    SVD_MODULE_AVAILABLE = False

try:
    from optimized_offensive_engine_v3 import (
        ProactiveIntelligence,
        AdvancedIDSEvasionSimulator,
        PPOEvasionAgent,
        DynamicIDS,
        AttackSurface,
        create_audit_dashboard,
        Config as OffensiveConfig
    )
    OFFENSIVE_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: optimized_offensive_engine_v3 non disponible: {e}")
    OFFENSIVE_MODULE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION GLOBALE
# ═══════════════════════════════════════════════════════════════════════════════

class AppConfig:
    """Configuration globale de l'application v3.0"""
    # Paramètres réseau
    N_NODES = 500
    CONNECTION_PROB = 0.06
    SEED = 42
    
    # Paramètres SVD
    K_DEFAULT = 30
    K_OPTIMAL = 20
    USE_SPARSE = True
    USE_GPU = False
    
    # Paramètres offensifs
    ATTACK_BUDGET = 0.25
    N_PERTURBATIONS = 200
    N_EVASION_EPISODES = 100
    
    # Paramètres IDS
    IDS_THRESHOLD = 2.5
    IDS_PCA_COMPONENTS = 5
    
    # Paramètres RL
    RL_LEARNING_RATE = 0.001
    RL_GAMMA = 0.99
    RL_HIDDEN_DIM = 64
    
    # Paramètres GAN
    GAN_LATENT_DIM = 10
    GAN_LR = 0.0002
    
    # Parallélisation
    N_JOBS = -1
    USE_PARALLEL = True
    
    # Visualisation
    DPI = 150
    FIGURE_SIZE = (24, 16)
    OUTPUT_DIR = './output'
    
    # Cache
    CACHE_L1_TTL = 60
    CACHE_L2_PATH = './cache_l2'
    
    # Performance
    PROFILE = True
    VERBOSE = True


# ═══════════════════════════════════════════════════════════════════════════════
# PROFILING DE PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

class PerformanceProfiler:
    """Profiler pour identifier les goulots d'étranglement en temps réel"""
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        self._start_times = {}
        
    def start(self, name):
        """Démarre le chronomètre pour une fonction"""
        self._start_times[name] = time.perf_counter()
        
    def stop(self, name):
        """Arrête le chronomètre et enregistre le temps"""
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)
            del self._start_times[name]
            
    def profile(self, func_name):
        """Décorateur de profiling"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                self.start(func_name)
                result = func(*args, **kwargs)
                self.stop(func_name)
                return result
            return wrapper
        return decorator
    
    def report(self):
        """Affiche le rapport de performance"""
        print("\\n" + "=" * 70)
        print("RAPPORT DE PERFORMANCE")
        print("=" * 70)
        print(f"{'Fonction':<40} {'Moyenne (ms)':<15} {'Max (ms)':<12} {'Appels':<8}")
        print("-" * 70)
        
        for name, times in sorted(self.timings.items(), key=lambda x: -np.mean(x[1])):
            mean_ms = np.mean(times) * 1000
            max_ms = np.max(times) * 1000
            n_calls = len(times)
            print(f"{name:<40} {mean_ms:>10.2f}     {max_ms:>8.2f}    {n_calls:>4d}")
        
        print("=" * 70)
        
    def get_summary(self):
        """Retourne un résumé des statistiques"""
        return {
            name: {
                'mean_ms': np.mean(times) * 1000,
                'max_ms': np.max(times) * 1000,
                'min_ms': np.min(times) * 1000,
                'total_ms': np.sum(times) * 1000,
                'calls': len(times)
            }
            for name, times in self.timings.items()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSE VISUALIZER UNIFIÉE
# ═══════════════════════════════════════════════════════════════════════════════

class Visualizer:
    """Module de visualisation unifié avec support de sortie configurable"""
    
    @staticmethod
    def create_basic_dashboard(analyzer, perturbations, impacts, adj_reconstructed, 
                                G_reconstructed, variance, output_path='network_svd_dashboard.png'):
        """Génère le dashboard d'analyse de base"""
        try:
            from network_svd_analyzer_v3 import create_dashboard as base_create_dashboard
            base_create_dashboard(analyzer, perturbations, impacts, adj_reconstructed, 
                                   G_reconstructed, variance, output_path=output_path)
        except ImportError:
            print(f"Warning: Visualisation de base non disponible")
            
    @staticmethod
    def create_offensive_dashboard(engine, surfaces, evasion_results, transfer, 
                                    output_path='audit_dashboard.png'):
        """Génère le dashboard offensif"""
        try:
            from optimized_offensive_engine_v3 import create_audit_dashboard
            create_audit_dashboard(engine, surfaces, evasion_results, transfer, 
                                    output_path=output_path)
        except ImportError:
            print(f"Warning: Visualisation offensive non disponible")


# ═══════════════════════════════════════════════════════════════════════════════
# MODES D'EXÉCUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_basic_mode(config, profiler):
    """Mode d'analyse de base (SVD/PCA standard)"""
    print("\\n" + "=" * 80)
    print("MODE ANALYSE DE BASE (SVD/PCA Standard)")
    print("=" * 80)
    
    # Créer le répertoire de sortie
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Initialisation du moteur SVD
    profiler.start("Initialisation Moteur")
    
    if SVD_MODULE_AVAILABLE:
        engine = OptimizedSVDEngine(
            n_nodes=min(config.N_NODES, 100),  # Limiter pour mode basic
            connection_prob=0.12,
            seed=config.SEED,
            use_sparse=False
        )
    else:
        print("Module SVD non disponible, utilisation du fallback")
        return
        
    profiler.stop("Initialisation Moteur")
    
    print(f"\\nRéseau généré: {engine.n_nodes} nœuds, {engine.G.number_of_edges()} arêtes")
    
    # SVD
    profiler.start("Décomposition SVD")
    U, S, Vt = engine.compute_svd(k=20, matrix_type='adjacency')
    profiler.stop("Décomposition SVD")
    
    print(f"\\nDécomposition SVD:")
    print(f"  σ₁ = {S[0]:.2f}, σ₅ = {S[4]:.2f}, σ₂₀ = {S[-1]:.2f}")
    print(f"  Variance expliquée par 10 composantes: {np.sum(S[:10]**2)/np.sum(S**2)*100:.1f}%")
    
    # Reconstruction
    profiler.start("Reconstruction Matrice")
    adj_recon, metrics = engine.reconstruct_matrix(k=10)
    profiler.stop("Reconstruction Matrice")
    
    print(f"\\nReconstruction:")
    print(f"  Erreur relative: {metrics['relative_error']*100:.2f}%")
    print(f"  Variance expliquée: {metrics['variance_explained']*100:.2f}%")
    
    # Visualisation
    if config.VERBOSE:
        output_path = os.path.join(config.OUTPUT_DIR, 'network_svd_dashboard.png')
        Visualizer.create_basic_dashboard(
            engine, [], [], adj_recon, engine.G, metrics['variance_explained'],
            output_path=output_path
        )
    
    print("\\n" + "=" * 60)
    print("ANALYSE DE BASE TERMINÉE")
    print("=" * 60)


def run_offensive_mode(config, profiler):
    """Mode d'analyse offensive (intelligence proactive)"""
    print("\\n" + "=" * 80)
    print("MODE INTELLIGENCE OFFENSIVE v3.0")
    print("=" * 80)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 1. Initialisation du moteur SVD optimisé
    profiler.start("Initialisation Moteur SVD")
    
    if not SVD_MODULE_AVAILABLE:
        print("Module SVD non disponible")
        return
        
    engine = OptimizedSVDEngine(
        n_nodes=config.N_NODES,
        connection_prob=config.CONNECTION_PROB,
        seed=config.SEED,
        use_sparse=config.USE_SPARSE
    )
    profiler.stop("Initialisation Moteur SVD")
    
    print(f"\\nRéseau: {engine.n_nodes} nœuds, {engine.G.number_of_edges()} arêtes")
    print(f"Cache L3 préchargé: {len(engine.cache.l3)} centralités")
    
    # 2. Fast SVD avec cache
    profiler.start("Fast SVD")
    U, S, Vt = engine.compute_svd(k=min(config.N_NODES//4, 50))
    profiler.stop("Fast SVD")
    
    # 3. Caractéristiques spectrales
    profiler.start("Caractéristiques Spectrales")
    features = engine.extract_spectral_features()
    profiler.stop("Caractéristiques Spectrales")
    
    print(f"\\nCaractéristiques spectrales:")
    for k, v in features.items():
        if not isinstance(v, np.ndarray):
            print(f"  {k}: {v:.4f}")
    
    # 4. Scan proactif de vulnérabilités
    if not OFFENSIVE_MODULE_AVAILABLE:
        print("Module offensif non disponible")
        return
        
    intelligence = ProactiveIntelligence(engine)
    
    profiler.start("Scan Proactif Vulnérabilités")
    surfaces = intelligence.scan_vulnerabilities(use_parallel=config.USE_PARALLEL)
    profiler.stop("Scan Proactif Vulnérabilités")
    
    print(f"\\nTop 10 Surfaces d'Attaque:")
    for i, s in enumerate(surfaces[:10]):
        print(f"  {i+1}. N{s.node_id}: {s.attack_vector} (Score: {s.exploitability_score:.3f}, "
              f"Détection: {s.detection_probability:.3f})")
    
    # 5. Évasion IDS Adaptative avec PPO
    simulator = AdvancedIDSEvasionSimulator(engine, intelligence)
    
    profiler.start("Évasion IDS PPO")
    evasion = simulator.simulate_adaptive_evasion(surfaces[0], n_episodes=config.N_EVASION_EPISODES)
    profiler.stop("Évasion IDS PPO")
    
    print(f"\\nRésultats d'évasion (cible N{surfaces[0].node_id}):")
    print(f"  Taux de détection moyen: {np.mean(evasion['detection_rates'])*100:.1f}%")
    print(f"  Taux de succès d'évasion: {np.mean(evasion['evasion_success'])*100:.1f}%")
    print(f"  Récompense moyenne PPO: {np.mean(evasion.get('rewards', [0])):.2f}")
    print(f"  Magnitude moyenne: {np.mean(evasion['perturbation_magnitudes']):.4f}")
    
    # 6. Transfer Learning
    transfer = {}
    if len(surfaces) >= 2:
        profiler.start("Transfer Learning")
        transfer = simulator.simulate_transfer_attack(surfaces[0], surfaces[1])
        profiler.stop("Transfer Learning")
        
        print(f"\\nTransfer Learning (N{surfaces[0].node_id} -> N{surfaces[1].node_id}):")
        print(f"  Détection source: {transfer['source_detection']:.4f}")
        print(f"  Détection cible: {transfer['target_detection']:.4f}")
        print(f"  Efficacité: {transfer['transfer_efficiency']:.4f}")
    
    # 7. Visualisation
    if config.VERBOSE:
        output_path = os.path.join(config.OUTPUT_DIR, 'audit_dashboard.png')
        Visualizer.create_offensive_dashboard(engine, surfaces, evasion, transfer, 
                                               output_path=output_path)
    
    # 8. Stats cache
    cache_stats = engine.cache.get_stats()
    print(f"\\nCache: {cache_stats['hit_rate']*100:.1f}% hit rate, "
          f"{cache_stats['l1_size']} entrées L1, {cache_stats['l3_size']} précalculs L3")
    
    print("\\n" + "=" * 60)
    print("INTELLIGENCE OFFENSIVE TERMINÉE")
    print("=" * 60)


def run_full_mode(config, profiler):
    """Mode complet (analyse de base + offensive en séquence)"""
    print("\\n" + "=" * 80)
    print("MODE COMPLET (Basic + Offensive)")
    print("=" * 80)
    
    run_basic_mode(config, profiler)
    run_offensive_mode(config, profiler)


def run_benchmark_mode(config, profiler):
    """Mode benchmark pour mesurer les performances"""
    print("\\n" + "=" * 80)
    print("MODE BENCHMARK")
    print("=" * 80)
    
    node_counts = [50, 100, 150, 300, 500, 1000]
    results = []
    
    for n_nodes in node_counts:
        if not SVD_MODULE_AVAILABLE:
            break
            
        print(f"\\nBenchmark avec {n_nodes} nœuds...")
        
        # Initialisation
        t0 = time.perf_counter()
        engine = OptimizedSVDEngine(n_nodes=n_nodes, seed=config.SEED, use_sparse=n_nodes > 100)
        init_time = (time.perf_counter() - t0) * 1000
        
        # SVD
        t0 = time.perf_counter()
        U, S, Vt = engine.compute_svd(k=min(n_nodes//4, 50))
        svd_time = (time.perf_counter() - t0) * 1000
        
        # Cache hit
        t0 = time.perf_counter()
        U, S, Vt = engine.compute_svd(k=min(n_nodes//4, 50))
        cache_time = (time.perf_counter() - t0) * 1000
        
        results.append({
            'nodes': n_nodes,
            'edges': engine.G.number_of_edges(),
            'init_ms': init_time,
            'svd_ms': svd_time,
            'cache_ms': cache_time,
            'speedup': svd_time / max(cache_time, 0.001)
        })
        
        print(f"  Init: {init_time:.1f}ms, SVD: {svd_time:.1f}ms, Cache: {cache_time:.1f}ms, "
              f"Speedup: {svd_time/max(cache_time, 0.001):.1f}x")
    
    # Résumé
    print("\\n" + "=" * 70)
    print("RÉSULTATS BENCHMARK")
    print("=" * 70)
    print(f"{'Nœuds':<10} {'Arêtes':<10} {'Init (ms)':<12} {'SVD (ms)':<12} {'Cache (ms)':<12} {'Speedup':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['nodes']:<10} {r['edges']:<10} {r['init_ms']:<12.1f} "
              f"{r['svd_ms']:<12.1f} {r['cache_ms']:<12.1f} {r['speedup']:<10.1f}x")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Point d'entrée principal de l'application v3.0"""
    parser = argparse.ArgumentParser(
        description='Network Security Analyzer v3.0 - SVD/PCA Optimisé',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python main_application_v3.py --mode basic --nodes 50
  python main_application_v3.py --mode offensive --nodes 500 --parallel
  python main_application_v3.py --mode full --nodes 1000 --gpu --verbose
  python main_application_v3.py --mode benchmark
        """
    )
    
    # Modes d'exécution
    parser.add_argument('--mode', choices=['basic', 'offensive', 'full', 'benchmark'],
                        default='full', help='Mode d\'exécution')
    
    # Paramètres réseau
    parser.add_argument('--nodes', type=int, default=500,
                        help='Nombre de nœuds du réseau (défaut: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed pour la reproductibilité')
    
    # Paramètres SVD
    parser.add_argument('--k-svd', type=int, default=30,
                        help='Nombre de composantes SVD (défaut: 30)')
    parser.add_argument('--sparse', action='store_true',
                        help='Utiliser SVD sparse pour grands réseaux')
    parser.add_argument('--gpu', action='store_true',
                        help='Utiliser GPU (CuPy) si disponible')
    
    # Paramètres offensifs
    parser.add_argument('--evasion-episodes', type=int, default=100,
                        help='Nombre d\'épisodes d\'évasion (défaut: 100)')
    parser.add_argument('--attack-budget', type=float, default=0.25,
                        help='Budget d\'attaque epsilon (défaut: 0.25)')
    
    # Paramètres performance
    parser.add_argument('--parallel', action='store_true',
                        help='Activer la parallélisation Joblib')
    parser.add_argument('--no-profile', action='store_true',
                        help='Désactiver le profiling')
    parser.add_argument('--no-verbose', action='store_true',
                        help='Mode silencieux')
    
    # Sortie
    parser.add_argument('--output', type=str, default='./output',
                        help='Répertoire de sortie')
    
    args = parser.parse_args()
    
    # Configuration
    config = AppConfig()
    config.N_NODES = args.nodes
    config.SEED = args.seed
    config.K_SVD = args.k_svd
    config.USE_SPARSE = args.sparse or args.nodes > 100
    config.USE_GPU = args.gpu
    config.N_EVASION_EPISODES = args.evasion_episodes
    config.ATTACK_BUDGET = args.attack_budget
    config.USE_PARALLEL = args.parallel
    config.PROFILE = not args.no_profile
    config.VERBOSE = not args.no_verbose
    config.OUTPUT_DIR = args.output
    
    # Affichage de la bannière
    print("=" * 80)
    print("NETWORK SECURITY ANALYZER v3.0 - SVD/PCA Optimisé")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Nœuds: {args.nodes}")
    print(f"Seed: {args.seed}")
    print(f"SVD: k={args.k_svd}, sparse={config.USE_SPARSE}, gpu={config.USE_GPU}")
    print(f"Parallélisation: {config.USE_PARALLEL}")
    print(f"Output: {args.output}")
    print("=" * 80)
    
    # Vérification des modules
    print("\\nModules disponibles:")
    print(f"  SVD/PCA: {'✓' if SVD_MODULE_AVAILABLE else '✗'}")
    print(f"  Offensif: {'✓' if OFFENSIVE_MODULE_AVAILABLE else '✗'}")
    
    # Initialisation du profiler
    profiler = PerformanceProfiler() if config.PROFILE else None
    
    # Exécution selon le mode
    try:
        if args.mode == 'basic':
            run_basic_mode(config, profiler)
        elif args.mode == 'offensive':
            run_offensive_mode(config, profiler)
        elif args.mode == 'benchmark':
            run_benchmark_mode(config, profiler)
        else:
            run_full_mode(config, profiler)
    except Exception as e:
        print(f"\\nErreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Rapport de performance
    if profiler and config.PROFILE:
        profiler.report()
    
    print("\\n" + "=" * 80)
    print("APPLICATION TERMINÉE")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

with open('/mnt/agents/output/main_application.py', 'w', encoding='utf-8') as f:
    f.write(code_v3)

print("main_application.py créé: {} caractères".format(len(code_v3)))
