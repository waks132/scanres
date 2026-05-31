"""
Network Security Analyzer v3.0 - Application Complète Finalisée
=================================================================
Application unifiée d'analyse de réseau par SVD/PCA avec toutes les corrections:

Corrections v3.0 (Audit Diff):
- A1: Reward normalisée par degré local, état normalisé par sous-graphe ego
- A2: IDS adversarial-robuste avec replay buffer + boucle co-évolution
- A3: Seuillage SVD au 90ème percentile (déjà corrigé dans network_svd_analyzer)
- A4: USE_PARALLEL propagé dans run_basic_mode et run_offensive_mode
- A5: JSON logging structuré + Dashboard Plotly HTML interactif
- Bug transfer_efficiency: src_det/tgt_det (pas tgt_det/src_det)

Optimisations implémentées:
- SVD Randomized (Halko et al. 2011) - O(nk²) vs O(n³)
- Cache Hiérarchique L1/L2/L3 (RAM, Disque, Précalculs)
- Détection Non-Linéaire (Kernel PCA + Isolation Forest + LOF)
- Parallélisation Joblib pour scan de vulnérabilités
- Évasion par Apprentissage par Renforcement (PPO) — A1 corrigé
- IDS Dynamique avec Online Learning (ADWIN) — A2 corrigé
- Transfer Learning avancé (Procrustes)
- Support GPU (CuPy - optionnel)

Modules:
    network_svd_analyzer.py      - Moteur SVD/PCA optimisé (v3.0)
    optimized_offensive_engine.py - Intelligence offensive proactive (v3.0)
    main_application.py          - Interface unifiée (ce fichier)

Usage:
    python main_application.py --mode basic      # Analyse de base
    python main_application.py --mode offensive  # Mode offensif
    python main_application.py --mode full       # Mode complet
    python main_application.py --nodes 1000 --gpu  # Grande échelle + GPU

Dépendances:
    numpy, scipy, matplotlib, seaborn, scikit-learn, networkx
    Optionnel: numba, joblib, cupy, torch, river, plotly

Auteur: Optimized by AI
Date: 2026-05-31
"""

import argparse
import sys
import os
import numpy as np
import time
import functools
import json
from datetime import datetime
from collections import OrderedDict

# Ajouter le répertoire courant au path pour les imports locaux
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
# A5 — LOGGING STRUCTURÉ JSON
# ═══════════════════════════════════════════════════════════════════════════════

class StructuredLogger:
    """Logger structuré JSON pour observabilité (A5)"""
    
    def __init__(self, name="nsa", output_dir="./output"):
        self.name = name
        self.output_dir = output_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(output_dir, f"run_{self.run_id}.jsonl")
        self.records = []
        os.makedirs(output_dir, exist_ok=True)
        
    def log(self, event, level="INFO", **kwargs):
        """Enregistre un événement structuré"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "level": level,
            "event": event,
            **kwargs
        }
        self.records.append(record)
        # Écrire immédiatement en mode append
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record, default=str) + "\\n")
        return record
    
    def info(self, event, **kwargs):
        return self.log(event, "INFO", **kwargs)
    
    def warn(self, event, **kwargs):
        return self.log(event, "WARN", **kwargs)
    
    def error(self, event, **kwargs):
        return self.log(event, "ERROR", **kwargs)
    
    def metric(self, metric_name, value, **kwargs):
        return self.log("metric", "METRIC", metric_name=metric_name, value=value, **kwargs)
    
    def save_summary(self, data):
        """Sauvegarde un résumé JSON complet du run"""
        summary_path = os.path.join(self.output_dir, f"summary_{self.run_id}.json")
        with open(summary_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return summary_path
    
    def get_records(self):
        return self.records


# Imports des modules métiers optimisés
try:
    from network_svd_analyzer import (
        NetworkSVDAnalyzer, 
        HierarchicalCache as SVDHierarchicalCache,
        create_dashboard as base_create_dashboard
    )
    SVD_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: network_svd_analyzer non disponible: {e}")
    SVD_MODULE_AVAILABLE = False

try:
    from optimized_offensive_engine import (
        OptimizedSVDEngine,
        OffensiveIntelligenceEngine,
        AdvancedIDSEvasionSimulator,
        PPOEvasionAgent,
        DynamicIDS,
        ADWINDetector,
        NonlinearDetector,
        AttackSurface,
        create_audit_dashboard
    )
    OFFENSIVE_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: optimized_offensive_engine non disponible: {e}")
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
    
    # A5: Logging
    STRUCTURED_LOGGING = True


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
        """Décorateur de profiling (corrigé avec functools.wraps)"""
        def decorator(func):
            @functools.wraps(func)
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
# A5 — DASHBOARD PLOTLY HTML INTERACTIF
# ═══════════════════════════════════════════════════════════════════════════════

class PlotlyDashboard:
    """Dashboard interactif HTML autonome (A5) — aucun serveur requis"""
    
    def __init__(self, output_dir="./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def create_interactive_dashboard(self, profiler_summary, evasion_results=None, 
                                     transfer_results=None, cache_stats=None,
                                     output_path="dashboard_interactive.html"):
        """Génère un dashboard HTML interactif avec Plotly"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.express as px
        except ImportError:
            print("[A5] Plotly non disponible — dashboard HTML ignoré")
            return None
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                "Temps d'exécution par phase",
                "Taux d'évasion par épisode",
                "Seuil IDS dynamique",
                "Récompenses RL (lissées)",
                "Distribution des perturbations",
                "Métriques de cache"
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "histogram"}, {"type": "bar"}]
            ]
        )
        
        # Panel 1: Temps d'exécution
        if profiler_summary:
            names = list(profiler_summary.keys())[:10]
            means = [profiler_summary[n]['mean_ms'] for n in names]
            fig.add_trace(go.Bar(x=names, y=means, name="Temps (ms)", 
                                 marker_color='steelblue'), row=1, col=1)
        
        # Panel 2: Taux d'évasion
        if evasion_results and 'evasion_success' in evasion_results:
            episodes = list(range(len(evasion_results['evasion_success'])))
            # Lissage
            window = 10
            if len(evasion_results['evasion_success']) >= window:
                smoothed = np.convolve(evasion_results['evasion_success'], 
                                       np.ones(window)/window, mode='valid')
                fig.add_trace(go.Scatter(x=list(range(window-1, len(episodes))), 
                                         y=smoothed, mode='lines', name="Évasion (lissé)",
                                         line=dict(color='green')), row=1, col=2)
        
        # Panel 3: Seuil IDS
        if evasion_results and 'ids_threshold_history' in evasion_results:
            episodes = list(range(len(evasion_results['ids_threshold_history'])))
            fig.add_trace(go.Scatter(x=episodes, 
                                     y=evasion_results['ids_threshold_history'],
                                     mode='lines', name="Seuil IDS",
                                     fill='tozeroy', line=dict(color='purple')), row=1, col=3)
        
        # Panel 4: Récompenses RL
        if evasion_results and 'rl_rewards' in evasion_results:
            rewards = evasion_results['rl_rewards']
            if len(rewards) >= 10:
                smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
                fig.add_trace(go.Scatter(x=list(range(9, len(rewards))), 
                                         y=smoothed, mode='lines', name="Reward RL",
                                         line=dict(color='blue')), row=2, col=1)
        
        # Panel 5: Distribution perturbations
        if evasion_results and 'perturbation_magnitudes' in evasion_results:
            fig.add_trace(go.Histogram(x=evasion_results['perturbation_magnitudes'],
                                      name="Magnitudes", marker_color='orange'), row=2, col=2)
        
        # Panel 6: Cache stats
        if cache_stats:
            labels = ['L1 Hit Rate (%)', 'L1 Size', 'L2 Files', 'L3 Metrics']
            values = [
                cache_stats.get('l1_hit_rate', 0) * 100,
                cache_stats.get('l1_size', 0),
                cache_stats.get('l2_files', 0),
                cache_stats.get('l3_metrics', 0)
            ]
            fig.add_trace(go.Bar(x=labels, y=values, name="Cache", 
                                 marker_color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']), 
                          row=2, col=3)
        
        fig.update_layout(
            title_text="Network Security Analyzer v3.0 — Dashboard Interactif (A5)",
            title_font_size=20,
            height=900,
            showlegend=False,
            template="plotly_white"
        )
        
        output_path = os.path.join(self.output_dir, output_path)
        fig.write_html(output_path, include_plotlyjs='cdn')
        print(f"[A5] Dashboard interactif sauvegardé: {output_path}")
        return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSE VISUALIZER UNIFIÉE
# ═══════════════════════════════════════════════════════════════════════════════

class Visualizer:
    """Module de visualisation unifié avec support de sortie configurable"""
    
    @staticmethod
    def create_basic_dashboard(analyzer, perturbations, impacts, adj_reconstructed, 
                                G_reconstructed, variance, output_path='network_svd_dashboard.png'):
        """Génère le dashboard d'analyse de base"""
        if SVD_MODULE_AVAILABLE:
            try:
                base_create_dashboard(analyzer, perturbations, impacts, adj_reconstructed, 
                                       G_reconstructed, variance, output_path=output_path)
            except Exception as e:
                print(f"Warning: Erreur lors de la visualisation de base: {e}")
        else:
            print("Warning: Visualisation de base non disponible (module manquant)")
            
    @staticmethod
    def create_offensive_dashboard(engine, surfaces, evasion_results, transfer, 
                                    output_path='audit_dashboard.png'):
        """Génère le dashboard offensif"""
        if OFFENSIVE_MODULE_AVAILABLE:
            try:
                create_audit_dashboard(engine, surfaces, evasion_results, transfer, 
                                       output_path=output_path)
            except Exception as e:
                print(f"Warning: Erreur lors de la visualisation offensive: {e}")
        else:
            print("Warning: Visualisation offensive non disponible (module manquant)")


# ═══════════════════════════════════════════════════════════════════════════════
# MODES D'EXÉCUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_basic_mode(config, profiler, logger=None):
    """Mode d'analyse de base (SVD/PCA standard via NetworkSVDAnalyzer)"""
    print("\\n" + "=" * 80)
    print("MODE ANALYSE DE BASE (SVD/PCA Standard)")
    print("=" * 80)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    if not SVD_MODULE_AVAILABLE:
        print("Module SVD non disponible, abandon.")
        if logger:
            logger.error("module_missing", module="network_svd_analyzer")
        return
        
    profiler.start("Initialisation Moteur Base")
    analyzer = NetworkSVDAnalyzer(
        n_nodes=min(config.N_NODES, 100),  # Limiter pour mode basic
        connection_prob=0.12,
        seed=config.SEED,
        attack_budget=config.ATTACK_BUDGET,
        use_cache=True
    )
    profiler.stop("Initialisation Moteur Base")
    
    print(f"\\nRéseau généré: {analyzer.n_nodes} nœuds, {analyzer.G.number_of_edges()} arêtes")
    if logger:
        logger.info("network_generated", nodes=analyzer.n_nodes, 
                   edges=analyzer.G.number_of_edges(), mode="basic")
    
    # SVD
    profiler.start("Décomposition SVD")
    S, variance_ratio = analyzer.perform_svd_decomposition()
    profiler.stop("Décomposition SVD")
    
    print(f"\\nDécomposition SVD:")
    print(f"  σ₁ = {S[0]:.2f}, σ₅ = {S[4]:.2f}, σ₂₀ = {S[min(19, len(S)-1)]:.2f}")
    if variance_ratio is not None and len(variance_ratio) >= 10:
        print(f"  Variance expliquée par 10 composantes: {np.sum(variance_ratio[:10])*100:.1f}%")
    if logger:
        logger.metric("svd_sigma_1", float(S[0]))
        logger.metric("variance_10_components", float(np.sum(variance_ratio[:10])*100) if variance_ratio is not None else 0)
    
    # Vulnérabilités
    profiler.start("Analyse Vulnérabilités")
    vulnerabilities = analyzer.analyze_vulnerabilities()
    profiler.stop("Analyse Vulnérabilités")
    
    # Perturbations — A4 CORRECTION: propagation de USE_PARALLEL
    profiler.start("Modélisation Perturbations")
    # A4 FIX: utilise config.USE_PARALLEL au lieu de False hardcodé
    perturbations, impacts = analyzer.model_stochastic_perturbations(
        n_perturbations=config.N_PERTURBATIONS,
        parallel=config.USE_PARALLEL  # A4 CORRECTION
    )
    profiler.stop("Modélisation Perturbations")
    
    if logger:
        logger.metric("perturbation_mean_impact", float(np.mean(impacts)))
        logger.metric("perturbation_max_impact", float(np.max(impacts)))
    
    # Reconstruction (Ciblage 90% de variance au lieu de 85%)
    profiler.start("Reconstruction Matrice")
    if variance_ratio is not None:
        k_opt = np.argmax(np.cumsum(variance_ratio) > 0.90) + 1
        if k_opt == 1:  # Fallback si la première composante dépasse 90%
            k_opt = max(5, int(len(variance_ratio) * 0.3))
    else:
        k_opt = 5
    adj_recon, G_reconstructed, variance = analyzer.reconstruct_reduced_network(k=k_opt)
    profiler.stop("Reconstruction Matrice")
    
    if logger:
        logger.metric("reconstruction_k", k_opt)
        logger.metric("reconstruction_variance", float(variance))
    
    # Visualisation
    if config.VERBOSE:
        output_path = os.path.join(config.OUTPUT_DIR, 'network_svd_dashboard.png')
        Visualizer.create_basic_dashboard(
            analyzer, perturbations, impacts, adj_recon, G_reconstructed, variance,
            output_path=output_path
        )
    
    print("\\n" + "=" * 60)
    print("ANALYSE DE BASE TERMINÉE")
    print("=" * 60)
    
    return {
        'analyzer': analyzer,
        'perturbations': perturbations,
        'impacts': impacts,
        'adj_reconstructed': adj_recon,
        'variance': variance
    }


def run_offensive_mode(config, profiler, logger=None):
    """Mode d'analyse offensive (intelligence proactive via OptimizedSVDEngine)"""
    print("\\n" + "=" * 80)
    print("MODE INTELLIGENCE OFFENSIVE v3.0")
    print("=" * 80)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    if not OFFENSIVE_MODULE_AVAILABLE:
        print("Module offensif non disponible, abandon.")
        if logger:
            logger.error("module_missing", module="optimized_offensive_engine")
        return
        
    # 1. Initialisation du moteur SVD optimisé
    profiler.start("Initialisation Moteur SVD Optimisé")
    engine = OptimizedSVDEngine(
        n_nodes=config.N_NODES,
        connection_prob=config.CONNECTION_PROB,
        seed=config.SEED,
        use_sparse=config.USE_SPARSE
    )
    profiler.stop("Initialisation Moteur SVD Optimisé")
    
    print(f"\\nRéseau: {engine.n_nodes} nœuds, {engine.G.number_of_edges()} arêtes")
    if engine.cache:
        print(f"Cache L3 préchargé: {len(engine.cache.l3)} centralités")
    if logger:
        logger.info("network_generated", nodes=engine.n_nodes, 
                   edges=engine.G.number_of_edges(), mode="offensive")
    
    # 2. Fast SVD avec cache
    profiler.start("Fast SVD")
    U, S, Vt = engine.compute_fast_svd(k=min(config.N_NODES//4, 50))
    profiler.stop("Fast SVD")
    
    # 3. Caractéristiques spectrales
    profiler.start("Caractéristiques Spectrales")
    features = engine.extract_spectral_features()
    profiler.stop("Caractéristiques Spectrales")
    
    print(f"\\nCaractéristiques spectrales:")
    for k, v in features.items():
        if not isinstance(v, np.ndarray):
            print(f"  {k}: {v:.4f}")
    if logger:
        for k, v in features.items():
            if not isinstance(v, np.ndarray):
                logger.metric(f"spectral_{k}", float(v))
    
    # 4. Scan proactif de vulnérabilités — A4: propagation parallélisation
    intelligence = OffensiveIntelligenceEngine(engine)
    
    profiler.start("Scan Proactif Vulnérabilités")
    # A4 FIX: propagation de USE_PARALLEL
    surfaces = intelligence.proactive_vulnerability_scan(
        use_nonlinear=True,
        use_parallel=config.USE_PARALLEL  # A4 CORRECTION
    )
    profiler.stop("Scan Proactif Vulnérabilités")
    
    print(f"\\nTop 10 Surfaces d'Attaque:")
    for i, s in enumerate(surfaces[:10]):
        print(f"  {i+1}. N{s.node_id}: {s.attack_vector} (Score: {s.exploitability_score:.3f}, "
              f"Détection: {s.detection_probability:.3f})")
    if logger:
        logger.metric("top_attack_score", float(surfaces[0].exploitability_score))
        logger.metric("attack_surfaces_count", len(surfaces))
    
    # 5. Évasion IDS Adaptative avec PPO — A1 corrigé
    simulator = AdvancedIDSEvasionSimulator(engine, intelligence)
    
    profiler.start("Évasion IDS PPO")
    evasion = simulator.simulate_adaptive_evasion(
        surfaces[0], 
        n_episodes=config.N_EVASION_EPISODES, 
        use_rl=True
    )
    profiler.stop("Évasion IDS PPO")
    
    print(f"\\nRésultats d'évasion (cible N{surfaces[0].node_id}):")
    print(f"  Taux de détection moyen: {np.mean(evasion['detection_rates'])*100:.1f}%")
    print(f"  Taux de succès d'évasion: {np.mean(evasion['evasion_success'])*100:.1f}%")
    if evasion.get('rl_rewards'):
        print(f"  Récompense moyenne PPO: {np.mean(evasion['rl_rewards']):.2f}")
    print(f"  Magnitude moyenne: {np.mean(evasion['perturbation_magnitudes']):.4f}")
    if logger:
        logger.metric("evasion_success_rate", float(np.mean(evasion['evasion_success'])))
        logger.metric("detection_rate_mean", float(np.mean(evasion['detection_rates'])))
        logger.metric("rl_reward_mean", float(np.mean(evasion['rl_rewards'])) if evasion.get('rl_rewards') else 0)
    
    # 6. Transfer Learning — BUG FIXÉ
    transfer = {}
    if len(surfaces) >= 2:
        profiler.start("Transfer Learning")
        transfer = simulator.simulate_transfer_attack(surfaces[0], surfaces[1])
        profiler.stop("Transfer Learning")
        
        print(f"\\nTransfer Learning (N{surfaces[0].node_id} -> N{surfaces[1].node_id}):")
        print(f"  Détection source: {transfer['source_detection']:.4f}")
        print(f"  Détection cible: {transfer['target_detection']:.4f}")
        print(f"  Efficacité: {transfer['transfer_efficiency']:.4f}")
        # A1: transfer_efficiency corrigé = src_det / tgt_det
        # Si > 1: transfert réussi (détection cible < source)
        # Si < 1: transfert échoué (détection cible > source)
        if logger:
            logger.metric("transfer_efficiency", float(transfer['transfer_efficiency']))
            logger.metric("transfer_source_detection", float(transfer['source_detection']))
            logger.metric("transfer_target_detection", float(transfer['target_detection']))
    
    # 7. Visualisation
    if config.VERBOSE:
        output_path = os.path.join(config.OUTPUT_DIR, 'audit_dashboard.png')
        Visualizer.create_offensive_dashboard(engine, surfaces, evasion, transfer, 
                                               output_path=output_path)
    
    # 8. Stats cache
    cache_stats = {}
    if engine.cache:
        cache_stats = engine.cache.stats()
        print(f"\\nCache: {cache_stats['l1_hit_rate']*100:.1f}% hit rate, "
              f"{cache_stats['l1_size']} entrées L1, {cache_stats['l3_metrics']} précalculs L3")
        if logger:
            logger.metric("cache_l1_hit_rate", float(cache_stats['l1_hit_rate']))
            logger.metric("cache_l1_size", cache_stats['l1_size'])
    
    print("\\n" + "=" * 60)
    print("INTELLIGENCE OFFENSIVE TERMINÉE")
    print("=" * 60)
    
    return {
        'engine': engine,
        'surfaces': surfaces,
        'evasion': evasion,
        'transfer': transfer,
        'cache_stats': cache_stats,
        'features': features
    }


def run_full_mode(config, profiler, logger=None):
    """Mode complet (analyse de base + offensive en séquence)"""
    print("\\n" + "=" * 80)
    print("MODE COMPLET (Basic + Offensive)")
    print("=" * 80)
    
    basic_results = run_basic_mode(config, profiler, logger)
    offensive_results = run_offensive_mode(config, profiler, logger)
    
    return {'basic': basic_results, 'offensive': offensive_results}


def run_benchmark_mode(config, profiler, logger=None):
    """Mode benchmark pour mesurer les performances"""
    print("\\n" + "=" * 80)
    print("MODE BENCHMARK")
    print("=" * 80)
    
    if not OFFENSIVE_MODULE_AVAILABLE:
        print("Module OptimizedSVDEngine requis pour le benchmark.")
        return
        
    node_counts = [50, 100, 150, 300, 500, 1000]
    results = []
    
    for n_nodes in node_counts:
        print(f"\\nBenchmark avec {n_nodes} nœuds...")
        
        # Initialisation
        t0 = time.perf_counter()
        engine = OptimizedSVDEngine(n_nodes=n_nodes, seed=config.SEED, use_sparse=n_nodes > 100)
        init_time = (time.perf_counter() - t0) * 1000
        
        # SVD (froid)
        t0 = time.perf_counter()
        U, S, Vt = engine.compute_fast_svd(k=min(n_nodes//4, 50), use_cache=False)
        svd_time = (time.perf_counter() - t0) * 1000
        
        # SVD (cache L1/L2)
        t0 = time.perf_counter()
        U, S, Vt = engine.compute_fast_svd(k=min(n_nodes//4, 50), use_cache=True)
        cache_time = (time.perf_counter() - t0) * 1000
        
        result = {
            'nodes': n_nodes,
            'edges': engine.G.number_of_edges(),
            'init_ms': init_time,
            'svd_ms': svd_time,
            'cache_ms': cache_time,
            'speedup': svd_time / max(cache_time, 0.001)
        }
        results.append(result)
        
        print(f"  Init: {init_time:.1f}ms, SVD: {svd_time:.1f}ms, Cache: {cache_time:.1f}ms, "
              f"Speedup: {svd_time/max(cache_time, 0.001):.1f}x")
        
        if logger:
            logger.metric("benchmark_init_ms", init_time, nodes=n_nodes)
            logger.metric("benchmark_svd_ms", svd_time, nodes=n_nodes)
            logger.metric("benchmark_cache_ms", cache_time, nodes=n_nodes)
            logger.metric("benchmark_speedup", svd_time/max(cache_time, 0.001), nodes=n_nodes)
    
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
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# A5 — SYSTÈME DE RÉGRESSION AUTOMATIQUE
# ═══════════════════════════════════════════════════════════════════════════════

class RegressionTester:
    """Système de régression automatique pour comparer les runs"""
    
    BASELINE = {
        "evasion_rate": 0.97,
        "frobenius_error": 28.41,
        "scan_time_ms": 3541,
        "transfer_efficiency_min": 0.5  # A1: doit être > 0.5 pour un transfert valide
    }
    
    def __init__(self, tolerance=0.05):
        self.tolerance = tolerance
        self.violations = []
    
    def check(self, metric, value, baseline_key=None):
        """Vérifie une métrique contre la baseline"""
        key = baseline_key or metric
        if key not in self.BASELINE:
            return True
        
        baseline_val = self.BASELINE[key]
        if isinstance(baseline_val, dict):
            # Range check
            if 'min' in baseline_val and value < baseline_val['min']:
                self.violations.append(f"{metric}: {value:.4f} < min {baseline_val['min']}")
                return False
            if 'max' in baseline_val and value > baseline_val['max']:
                self.violations.append(f"{metric}: {value:.4f} > max {baseline_val['max']}")
                return False
        else:
            delta = abs(value - baseline_val) / baseline_val
            if delta > self.tolerance:
                self.violations.append(f"{metric}: {value:.4f} vs baseline {baseline_val:.4f} (delta {delta:.1%})")
                return False
        return True
    
    def report(self):
        """Affiche le rapport de régression"""
        print("\\n" + "=" * 70)
        print("RAPPORT DE RÉGRESSION AUTOMATIQUE (A5)")
        print("=" * 70)
        if not self.violations:
            print("✓ Aucune régression détectée")
        else:
            print(f"✗ {len(self.violations)} régression(s) détectée(s):")
            for v in self.violations:
                print(f"  • {v}")
        print("=" * 70)
        return len(self.violations) == 0


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
  python main_application.py --mode basic --nodes 50
  python main_application.py --mode offensive --nodes 500 --parallel
  python main_application.py --mode full --nodes 1000 --verbose
  python main_application.py --mode benchmark
        """
    )
    
    # Modes d'exécution
    parser.add_argument('--mode', choices=['basic', 'offensive', 'full', 'benchmark'],
                        default='full', help="Mode d'exécution")
    
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
                        help="Nombre d'épisodes d'évasion (défaut: 100)")
    parser.add_argument('--attack-budget', type=float, default=0.25,
                        help="Budget d'attaque epsilon (défaut: 0.25)")
    
    # Paramètres performance
    parser.add_argument('--parallel', action='store_true',
                        help='Activer la parallélisation Joblib')
    parser.add_argument('--no-profile', action='store_true',
                        help='Désactiver le profiling')
    parser.add_argument('--no-verbose', action='store_true',
                        help='Mode silencieux')
    
    # A5: Paramètres logging
    parser.add_argument('--no-json-log', action='store_true',
                        help='Désactiver le logging JSON structuré')
    parser.add_argument('--no-plotly', action='store_true',
                        help='Désactiver le dashboard Plotly HTML')
    parser.add_argument('--regression-test', action='store_true',
                        help='Activer les tests de régression automatiques')
    
    # Sortie
    parser.add_argument('--output', type=str, default='./output',
                        help='Répertoire de sortie')
    
    args = parser.parse_args()
    
    # Configuration dynamique
    config = AppConfig()
    config.N_NODES = args.nodes
    config.SEED = args.seed
    config.K_DEFAULT = args.k_svd
    config.USE_SPARSE = args.sparse or args.nodes > 100
    config.USE_GPU = args.gpu
    config.N_EVASION_EPISODES = args.evasion_episodes
    config.ATTACK_BUDGET = args.attack_budget
    config.USE_PARALLEL = args.parallel  # A4: propagé dans les modes
    config.PROFILE = not args.no_profile
    config.VERBOSE = not args.no_verbose
    config.OUTPUT_DIR = args.output
    config.STRUCTURED_LOGGING = not args.no_json_log
    
    # A5: Initialiser le logger structuré
    logger = None
    if config.STRUCTURED_LOGGING:
        logger = StructuredLogger(name="nsa", output_dir=config.OUTPUT_DIR)
        logger.info("application_start", 
                   mode=args.mode, nodes=args.nodes, seed=args.seed,
                   parallel=config.USE_PARALLEL, version="3.0")
    
    # Affichage de la bannière
    print("=" * 80)
    print("NETWORK SECURITY ANALYZER v3.0 - SVD/PCA Optimisé")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Nœuds: {args.nodes}")
    print(f"Seed: {args.seed}")
    print(f"SVD: k={args.k_svd}, sparse={config.USE_SPARSE}, gpu={config.USE_GPU}")
    print(f"Parallélisation: {config.USE_PARALLEL}")  # A4: affichage
    print(f"Output: {args.output}")
    print("=" * 80)
    
    # Vérification des modules
    print("\\nModules disponibles:")
    print(f"  SVD/PCA (Base): {'✓' if SVD_MODULE_AVAILABLE else '✗'}")
    print(f"  Moteur Offensif (Optimisé): {'✓' if OFFENSIVE_MODULE_AVAILABLE else '✗'}")
    if logger:
        logger.info("modules_status", 
                   svd=SVD_MODULE_AVAILABLE, offensive=OFFENSIVE_MODULE_AVAILABLE)
    
    # Initialisation du profiler
    profiler = PerformanceProfiler() if config.PROFILE else None
    
    # Exécution selon le mode
    results = {}
    try:
        if args.mode == 'basic':
            results = run_basic_mode(config, profiler, logger)
        elif args.mode == 'offensive':
            results = run_offensive_mode(config, profiler, logger)
        elif args.mode == 'benchmark':
            results = run_benchmark_mode(config, profiler, logger)
        else:
            results = run_full_mode(config, profiler, logger)
            
        if logger:
            logger.info("execution_complete", mode=args.mode, status="success")
            
    except Exception as e:
        print(f"\\nErreur lors de l'exécution: {e}")
        if logger:
            logger.error("execution_failed", error=str(e), error_type=type(e).__name__)
        import traceback
        traceback.print_exc()
        return 1
    
    # Rapport de performance
    profiler_summary = {}
    if profiler and config.PROFILE:
        profiler.report()
        profiler_summary = profiler.get_summary()
        if logger:
            for name, stats in profiler_summary.items():
                logger.metric(f"perf_{name}_ms", stats['mean_ms'])
    
    # A5: Dashboard Plotly interactif
    if not args.no_plotly and (args.mode in ['offensive', 'full']):
        try:
            dashboard = PlotlyDashboard(output_dir=config.OUTPUT_DIR)
            evasion = results.get('evasion') if isinstance(results, dict) else None
            transfer = results.get('transfer') if isinstance(results, dict) else None
            cache_stats = results.get('cache_stats') if isinstance(results, dict) else None
            
            dashboard.create_interactive_dashboard(
                profiler_summary=profiler_summary,
                evasion_results=evasion,
                transfer_results=transfer,
                cache_stats=cache_stats
            )
        except Exception as e:
            print(f"[A5] Erreur dashboard Plotly: {e}")
    
    # A5: Tests de régression
    if args.regression_test and results:
        tester = RegressionTester()
        if isinstance(results, dict):
            if 'evasion' in results and results['evasion']:
                evasion_rate = np.mean(results['evasion']['evasion_success'])
                tester.check("evasion_rate", evasion_rate)
            if 'transfer' in results and results['transfer']:
                eff = results['transfer'].get('transfer_efficiency', 0)
                tester.check("transfer_efficiency", eff, "transfer_efficiency_min")
        tester.report()
    
    # A5: Sauvegarder le résumé JSON
    if logger:
        summary = {
            "run_id": logger.run_id,
            "mode": args.mode,
            "config": {
                "nodes": config.N_NODES,
                "seed": config.SEED,
                "parallel": config.USE_PARALLEL,
                "sparse": config.USE_SPARSE
            },
            "profiler": profiler_summary,
            "results": {
                k: str(v) if not isinstance(v, (int, float, bool, str, list, dict)) else v
                for k, v in (results.items() if isinstance(results, dict) else {})
            }
        }
        summary_path = logger.save_summary(summary)
        print(f"\\n[A5] Résumé JSON sauvegardé: {summary_path}")
        print(f"[A5] Logs structurés: {logger.log_file}")
    
    print("\\n" + "=" * 80)
    print("APPLICATION TERMINÉE")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


