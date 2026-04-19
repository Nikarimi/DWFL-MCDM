import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from scipy.stats import dirichlet, ttest_rel, ttest_ind, norm, sem
from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import os
import pymcdm
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
import copy
import concurrent.futures
from functools import lru_cache
import time
warnings.filterwarnings('ignore')

# تنظیمات پیشرفته GPU برای عملکرد بهتر
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # فعال کردن رشد پویای حافظه
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # تنظیمات بهینه برای عملکرد
        tf.config.optimizer.set_jit(True)  # فعال کردن XLA JIT compilation
        
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("Using CPU")

# تابع کمکی برای تبدیل انواع numpy به types استاندارد پایتون برای JSON
def convert_to_serializable(obj):
    """تبدیل انواع غیرقابل سریال‌سازی JSON به انواع استاندارد پایتون"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif hasattr(obj, 'tolist'):  # برای tensorflow tensors
        return obj.tolist()
    else:
        return obj

class JSONEncoder(json.JSONEncoder):
    """Encoder سفارشی برای هندل کردن انواع numpy"""
    def default(self, obj):
        return convert_to_serializable(obj)

@dataclass
class Config:
    """کلاس تنظیمات بهینه‌سازی شده"""
    dataset: str = 'mnist'  # 'mnist', 'fashion_mnist', 'cifar10'
    num_nodes: int = 10
    num_rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    mobility_model: str = 'rwp'  # rw, rwp, rd, rpgm, manhattan, freeway, map_based, uav_formation, uav_random_waypoint
    distance_method: str = 'dynamic'  # cosine, cka, dynamic
    alpha: float = 0.4  # برای ترکیب دینامیک
    use_gpu: bool = True
    save_dir: str = None
    seed: int = 42
    num_runs: int = 1  # تعداد اجراهای شبیه‌سازی
    
    # پارامترهای توزیع دیتا
    dirichlet_alpha: float = 100
    
    # پارامترهای تحرک
    area_size: int = 1000
    speed_range: tuple = (1, 5)
    pause_time: int = 1
    communication_range: float = 150
    
    # پارامترهای وزن‌دهی
    data_weight_param: float = 4.0  # α
    accuracy_weight_param: float = 1.0  # β
    accuracy_threshold: float = 0.3
    entropy_weight_param: float = 0.5  # γ
    
    # پارامترهای WAFL
    wafl_lambda: float = 0.5  # λ coefficient for model aggregation
    
    # پارامتر DFL-DGA
    dfl_beta: float = 0.8  # ضریب ترکیب وزن‌های قبلی با وزن‌های مجموعه برتری در DFL-DGA
    
    def create_save_dir(self):
        """ایجاد پوشه ذخیره نتایج"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"WAFLbased-ThreeMethodsResult-v2/multi_run_nodes{self.num_nodes}_rounds{self.num_rounds}_runs{self.num_runs}_dataset_{self.dataset}_alpha_{self.dirichlet_alpha}_dfl-beta_{self.dfl_beta}_{self.mobility_model}_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        return self.save_dir

class DataHandler:
    """مدیریت داده‌ها"""
    def __init__(self, config: Config):
        self.config = config
        self.load_dataset()
        # ذخیره توزیع داده‌ها برای استفاده ثابت در همه اجراها
        self.fixed_client_data = None
        self.fixed_client_labels = None
        
    def load_dataset(self):
        """بارگذاری دیتاست"""
        # تعریف اولیه متغیرها برای جلوگیری از UnboundLocalError
        x_train, y_train, x_test, y_test = None, None, None, None
        
        if self.config.dataset == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            self.num_classes = 10
            self.input_shape = (28, 28, 1)
            
        elif self.config.dataset == 'fashion_mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            self.num_classes = 10
            self.input_shape = (28, 28, 1)
            
        elif self.config.dataset == 'cifar10':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
            y_train = y_train.flatten()
            y_test = y_test.flatten()
            self.num_classes = 10
            self.input_shape = (32, 32, 3)
        else:
            raise ValueError(f"Dataset {self.config.dataset} not supported. Use 'mnist', 'fashion_mnist', or 'cifar10'")
        
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train
        
    def dirichlet_split(self, num_clients: int, alpha: float = 0.5, seed: int = None):
        """تقسیم داده با توزیع دیریکله"""
        if seed is not None:
            np.random.seed(seed)
            
        num_classes = self.num_classes
        idx = [np.where(self.y_train == i)[0] for i in range(num_classes)]
        
        client_data = [[] for _ in range(num_clients)]
        client_labels = [[] for _ in range(num_clients)]
        
        for k in range(num_classes):
            np.random.shuffle(idx[k])
            proportions = dirichlet.rvs([alpha] * num_clients).flatten()
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx[k])).astype(int)[:-1]
            
            idx_split = np.split(idx[k], proportions)
            
            for i in range(num_clients):
                client_data[i].extend(self.x_train[idx_split[i]])
                client_labels[i].extend(self.y_train[idx_split[i]])
                
        return client_data, client_labels
    
    def get_fixed_client_datasets(self, num_clients: int):
        """تهیه دیتاست ثابت برای همه اجراها"""
        if self.fixed_client_data is None or self.fixed_client_labels is None:
            # استفاده از seed ثابت برای تولید توزیع داده‌های یکسان
            self.fixed_client_data, self.fixed_client_labels = self.dirichlet_split(
                num_clients, self.config.dirichlet_alpha, self.config.seed
            )
        
        datasets = []
        data_stats = []
        
        for i in range(num_clients):
            data = np.array(self.fixed_client_data[i])
            labels = np.array(self.fixed_client_labels[i])
            
            # محاسبه آمار
            class_dist = np.bincount(labels, minlength=self.num_classes)
            if len(data) > 0:
                class_probs = class_dist / len(labels)
                entropy = -np.sum(class_probs * np.log2(class_probs + 1e-10))
            else:
                entropy = 0
            
            dataset = tf.data.Dataset.from_tensor_slices((data, labels))
            dataset = dataset.shuffle(len(data)).batch(self.config.batch_size)
            
            datasets.append(dataset)
            data_stats.append({
                'size': len(data),
                'class_distribution': class_dist.tolist(),
                'entropy': float(entropy)
            })
            
        return datasets, data_stats

class ModelHandler:
    """مدیریت مدل‌ها"""
    def __init__(self, config: Config, num_classes: int, input_shape: tuple):
        self.config = config
        self.num_classes = num_classes
        self.input_shape = input_shape
        
    def create_model(self):
        """ایجاد مدل بر اساس دیتاست"""
        if self.config.dataset == 'cifar10':
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(self.num_classes, activation='softmax')
            ])
        else:  # برای MNIST و Fashion MNIST
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(self.num_classes, activation='softmax')
            ])
            
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

class MobilityManager:
    """مدیریت تحرک گره‌ها"""
    def __init__(self, config: Config):
        self.config = config
        self.positions = []
        self.trajectories = []
        self.meeting_history = []
        self.destinations = None
        self.speeds = None
        
    def initialize_positions(self, num_nodes: int, seed: int = None):
        """مقداردهی اولیه موقعیت‌ها"""
        if seed is not None:
            np.random.seed(seed)
            
        area = self.config.area_size
        
        if self.config.mobility_model in ['uav_formation', 'uav_random_waypoint']:
            self.positions = np.random.rand(num_nodes, 3) * area
            self.positions[:, 2] = np.random.uniform(50, 200, num_nodes)
        else:
            self.positions = np.random.rand(num_nodes, 2) * area
            
        self.trajectories = [[pos.copy()] for pos in self.positions]
        self.meeting_history = []
        
        if self.config.mobility_model == 'rwp':
            self.destinations = np.random.rand(num_nodes, self.positions.shape[1]) * area
            self.speeds = np.random.uniform(*self.config.speed_range, num_nodes)
        
    def move_nodes(self, seed: int = None):
        """حرکت دادن گره‌ها بر اساس مدل تحرک"""
        if seed is not None:
            np.random.seed(seed)
            
        num_nodes = len(self.positions)
        area = self.config.area_size
        
        if self.config.mobility_model == 'rw':
            angles = np.random.rand(num_nodes) * 2 * np.pi
            speeds = np.random.uniform(*self.config.speed_range, num_nodes)
            
            if self.positions.shape[1] == 3:
                dx = speeds * np.cos(angles)
                dy = speeds * np.sin(angles)
                dz = np.random.uniform(-1, 1, num_nodes)
                self.positions += np.column_stack([dx, dy, dz])
            else:
                dx = speeds * np.cos(angles)
                dy = speeds * np.sin(angles)
                self.positions += np.column_stack([dx, dy])
                
        elif self.config.mobility_model == 'rwp':
            for i in range(num_nodes):
                direction = self.destinations[i] - self.positions[i]
                distance = np.linalg.norm(direction)
                
                if distance < 5:
                    self.destinations[i] = np.random.rand(self.positions.shape[1]) * area
                    self.speeds[i] = np.random.uniform(*self.config.speed_range)
                    direction = self.destinations[i] - self.positions[i]
                    distance = np.linalg.norm(direction)
                
                if distance > 0:
                    move = (direction / distance) * min(self.speeds[i], distance)
                    self.positions[i] += move
                    
        elif self.config.mobility_model == 'manhattan':
            directions = np.random.choice([0, 1, 2, 3], num_nodes)
            speeds = np.random.uniform(*self.config.speed_range, num_nodes)
            
            for i in range(num_nodes):
                if directions[i] == 0:
                    self.positions[i, 1] += speeds[i]
                elif directions[i] == 1:
                    self.positions[i, 0] += speeds[i]
                elif directions[i] == 2:
                    self.positions[i, 1] -= speeds[i]
                else:
                    self.positions[i, 0] -= speeds[i]
        
        self.positions = np.clip(self.positions, 0, area)
        
        for i in range(num_nodes):
            self.trajectories[i].append(self.positions[i].copy())
            
    def get_meetings(self):
        """تشخیص ملاقات‌ها بر اساس فاصله"""
        num_nodes = len(self.positions)
        meetings = []
        
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if self.positions.shape[1] == 3:
                    distance = np.linalg.norm(self.positions[i] - self.positions[j])
                else:
                    distance = np.linalg.norm(self.positions[i] - self.positions[j])
                    
                if distance < self.config.communication_range:
                    meetings.append((i, j, distance))
        
        self.meeting_history.append(meetings)
        return meetings

class GraphManager:
    """مدیریت گراف‌های محلی"""
    def __init__(self, config: Config):
        self.config = config
        self.distance_cache = {}
        
    def calculate_node_weight(self, data_size: float, accuracy: float, entropy: float) -> float:
        """محاسبه وزن گره با توابع غیرخطی"""
        normalized_data = min(data_size / 1000.0, 1.0) if data_size > 0 else 0
        
        f_data = np.tanh(self.config.data_weight_param * normalized_data)
        f_accuracy = 1.0 / (1.0 + (accuracy - self.config.accuracy_threshold))

        
        weight = f_data * f_accuracy 
        
        if data_size < 100 or accuracy < 0.8:
            weight *= 0.1
            
        return weight
    
    def calculate_distance(self, model1_weights, model2_weights, method: str = 'cosine') -> float:
        """محاسبه فاصله بین دو مدل"""
        if len(model1_weights) == 0 or len(model2_weights) == 0:
            return 1.0
        
        if method == 'cosine':
            dot_product = np.dot(model1_weights, model2_weights)
            norm1 = np.linalg.norm(model1_weights)
            norm2 = np.linalg.norm(model2_weights)
            
            if norm1 == 0 or norm2 == 0:
                return 1.0
                
            similarity = dot_product / (norm1 * norm2)
            distance = 1.0 - similarity
            
        elif method == 'pearson':
            w1_centered = model1_weights - np.mean(model1_weights)
            w2_centered = model2_weights - np.mean(model2_weights)
            
            covariance = np.dot(w1_centered, w2_centered)
            var1 = np.dot(w1_centered, w1_centered)
            var2 = np.dot(w2_centered, w2_centered)
            
            if var1 == 0 or var2 == 0:
                return 1.0
                
            similarity = covariance / (np.sqrt(var1) * np.sqrt(var2))
            distance = 1.0 - (similarity + 1) / 2
        else:
            cos_dist = self.calculate_distance(model1_weights, model2_weights, 'cosine')
            pearson_dist = self.calculate_distance(model1_weights, model2_weights, 'pearson')
            distance = self.config.alpha * cos_dist + (1 - self.config.alpha) * pearson_dist
            
        return max(0.0, min(1.0, distance))
    
    def extract_model_weights(self, model):
        """استخراج وزن‌های مدل"""
        weights = []
        for layer in model.layers[-3:]:
            if len(layer.get_weights()) > 0:
                weights.append(layer.get_weights()[0].flatten())
        return np.concatenate(weights) if weights else np.array([])
    
    def calculate_model_similarity(self, models: List) -> Dict:
        """محاسبه شباهت بین تمام مدل‌ها"""
        if len(models) < 2:
            return {'mean': 0.0, 'std': 0.0, 'pairwise_similarities': []}
        
        model_weights = [self.extract_model_weights(model) for model in models]
        similarities = []
        pairwise_details = []
        
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                distance = self.calculate_distance(
                    model_weights[i],
                    model_weights[j],
                    self.config.distance_method
                )
                similarity = 1.0 - distance
                similarities.append(similarity)
                pairwise_details.append({
                    'node_i': i,
                    'node_j': j,
                    'similarity': float(similarity)
                })
        
        if similarities:
            return {
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities)),
                'pairwise_similarities': pairwise_details
            }
        else:
            return {'mean': 0.0, 'std': 0.0, 'pairwise_similarities': []}

class DominatingSetSelector:
    """انتخاب مجموعه برتری"""
    def __init__(self, config: Config, graph_manager: GraphManager):
        self.config = config
        self.graph_manager = graph_manager
    
    def calculate_node_score(self, node: int, graph: nx.Graph, data_sizes: List[float], 
                           accuracies: List[float], models_weights: List) -> float:
        neighbors = list(graph.neighbors(node))
        
        if not neighbors:
            return accuracies[node] * data_sizes[node]
        
        total_weighted_distance = 0.0
        for neighbor in neighbors:
            if graph.has_edge(node, neighbor):
                distance = self.graph_manager.calculate_distance(
                    models_weights[node], models_weights[neighbor], self.config.distance_method
                )
                weight = graph.nodes[neighbor]['weight']
                total_weighted_distance += distance * weight
        
        score = (accuracies[node] * data_sizes[node]) / (1.0 + total_weighted_distance)
        return score
    
    def select_dominating_set(self, graph: nx.Graph, data_sizes: List[float], 
                            accuracies: List[float], models_weights: List) -> List[int]:
        dominating_set = set()
        covered = set()
        remaining_nodes = set(graph.nodes())
        
        while remaining_nodes - covered:
            scores = {}
            for node in remaining_nodes - covered:
                scores[node] = self.calculate_node_score(node, graph, data_sizes, accuracies, models_weights)
            
            if not scores:
                break
            
            best_node = max(scores.items(), key=lambda x: x[1])[0]
            dominating_set.add(best_node)
            covered.add(best_node)
            
            neighbors = set(graph.neighbors(best_node))
            covered.update(neighbors)
        
        return list(dominating_set)

class AHP_WASPAS:
    """روش AHP+WASPAS برای وزن‌دهی نهایی"""
    def __init__(self):
        pass
    
    def calculate_ahp_weights(self, criteria_comparison: np.ndarray) -> Tuple[np.ndarray, float]:
        n = criteria_comparison.shape[0]
        col_sums = criteria_comparison.sum(axis=0)
        normalized = criteria_comparison / col_sums
        weights = normalized.mean(axis=1)
        
        weighted_sum = criteria_comparison @ weights
        lambda_max = np.mean(weighted_sum / weights)
        ci = (lambda_max - n) / (n - 1)

        return weights
    
    def calculate_waspas_scores(self, decision_matrix: np.ndarray, criteria_weights: np.ndarray, 
                              criteria_types: List[str]) -> np.ndarray:
        m, n = decision_matrix.shape
        normalized = np.zeros((m, n))
        
        for j in range(n):
            if criteria_types[j] == 'max':
                col_max = decision_matrix[:, j].max()
                if col_max > 0:
                    normalized[:, j] = decision_matrix[:, j] / col_max
                else:
                    normalized[:, j] = 0
            else:
                col_min = decision_matrix[:, j].min()
                if col_min > 0:
                    normalized[:, j] = col_min / decision_matrix[:, j]
                else:
                    normalized[:, j] = 1.0
        
        scores = np.zeros(m)
        
        for i in range(m):
            wsm = np.sum(criteria_weights * normalized[i, :])
            wpm = 1.0
            for j in range(n):
                if normalized[i, j] > 0:
                    wpm *= normalized[i, j] ** criteria_weights[j]
            
            scores[i] = 0.5 * wsm + 0.5 * wpm
        
        score_sum = scores.sum()
        if score_sum > 0:
            scores = scores / score_sum
        
        return scores

class MobilityTrajectoryGenerator:
    """تولید کننده مسیرهای تحرک مشترک"""
    def __init__(self, config: Config):
        self.config = config
        self.mobility_manager = MobilityManager(config)
        
    def generate_trajectory(self, run_id: int):
        """تولید مسیر تحرک برای یک اجرا"""
        # مقداردهی اولیه موقعیت‌ها
        seed = self.config.seed + run_id * 1000
        self.mobility_manager.initialize_positions(self.config.num_nodes, seed)
        
        # ذخیره موقعیت اولیه
        positions_list = [self.mobility_manager.positions.copy()]
        
        # تولید مسیر برای همه دورها
        for round_num in range(self.config.num_rounds):
            # حرکت گره‌ها
            mobility_seed = self.config.seed + run_id * 1000 + round_num * 100
            self.mobility_manager.move_nodes(mobility_seed)
            
            # ذخیره موقعیت‌ها
            positions_list.append(self.mobility_manager.positions.copy())
        
        # محاسبه ملاقات‌ها برای هر دور
        meetings_list = []
        for positions in positions_list[1:]:  # از دور اول شروع می‌کنیم
            # ایجاد یک mobility manager موقت برای محاسبه ملاقات‌ها
            temp_manager = MobilityManager(self.config)
            temp_manager.positions = positions.copy()
            meetings = temp_manager.get_meetings()
            meetings_list.append(meetings)
        
        return positions_list, meetings_list

class DFL_DGA_WithSharedMobility:
    """DFL-DGA با مسیرهای تحرک مشترک"""
    def __init__(self, config: Config, run_id: int = 0, 
                 precomputed_positions: List[np.ndarray] = None,
                 precomputed_meetings: List[List[Tuple]] = None):
        self.config = config
        self.run_id = run_id
        self.precomputed_positions = precomputed_positions
        self.precomputed_meetings = precomputed_meetings
        
        self.data_handler = DataHandler(config)
        self.model_handler = ModelHandler(config, self.data_handler.num_classes, self.data_handler.input_shape)
        self.mobility_manager = MobilityManager(config)
        self.graph_manager = GraphManager(config)
        self.ds_selector = DominatingSetSelector(config, self.graph_manager)
        self.ahp_waspas = AHP_WASPAS()
        
        self.models = []
        self.client_datasets = []
        self.data_stats = []
        self.accuracies = []
        self.local_graphs = []
        
        self.results = {
            'accuracy': [],
            'f1_scores': [],
            'loss': [],
            'meetings_per_node': [],
            'model_similarities': [],
            'node_weights': [],
            'dominating_set_sizes': [],
            'individual_accuracies': [],
            'individual_losses': [],
            'positions_used': []  # ذخیره موقعیت‌های استفاده شده
        }
        
    def initialize(self, seed_offset: int = 0):
        # استفاده از توزیع داده ثابت
        self.client_datasets, self.data_stats = self.data_handler.get_fixed_client_datasets(
            self.config.num_nodes
        )
        
        self.models = [self.model_handler.create_model() for _ in range(self.config.num_nodes)]
        
        self.local_graphs = []
        for i in range(self.config.num_nodes):
            G = nx.Graph()
            G.add_node(i, weight=1.0)
            self.local_graphs.append(G)
        
        # مقداردهی اولیه موقعیت‌ها با اولین موقعیت از پیش محاسبه شده
        if self.precomputed_positions:
            self.mobility_manager.positions = self.precomputed_positions[0].copy()
        else:
            mobility_seed = self.config.seed + self.run_id * 1000 + seed_offset
            self.mobility_manager.initialize_positions(self.config.num_nodes, mobility_seed)
        
        self.accuracies = [0.0] * self.config.num_nodes
    
    def train_single_node(self, node_id: int):
        """آموزش یک گره"""
        model = self.models[node_id]
        dataset = self.client_datasets[node_id]
        
        # آموزش مدل
        history = model.fit(
            dataset,
            epochs=self.config.local_epochs,
            verbose=0
        )
        
        # پیش‌بینی
        predictions = model.predict(self.data_handler.x_test, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)
        
        accuracy = float(accuracy_score(self.data_handler.y_test, pred_labels))
        f1 = float(f1_score(self.data_handler.y_test, pred_labels, average='macro'))
        loss = float(history.history['loss'][-1])
        
        self.accuracies[node_id] = accuracy
        
        return loss, accuracy, f1, history.history
    
    def local_train_sequential(self):
        """آموزش ترتیبی همه گره‌ها"""
        losses = []
        accuracies = []
        f1_scores = []
        histories = []
        
        # اجرای ترتیبی
        for i in range(self.config.num_nodes):
            loss, acc, f1, history = self.train_single_node(i)
            losses.append(loss)
            accuracies.append(acc)
            f1_scores.append(f1)
            histories.append(history)
        
        return losses, accuracies, f1_scores, histories
    
    def run_round(self, round_num: int):
        """اجرای یک دور"""
        round_start_time = time.time()
        
        # آموزش ترتیبی
        round_losses, round_accuracies, round_f1_scores, round_histories = self.local_train_sequential()
        
        # استفاده از موقعیت‌های از پیش محاسبه شده
        if self.precomputed_positions and round_num < len(self.precomputed_positions) - 1:
            # موقعیت‌های از پیش محاسبه شده شامل موقعیت اولیه + موقعیت‌های هر دور است
            self.mobility_manager.positions = self.precomputed_positions[round_num + 1].copy()
            self.results['positions_used'].append(self.precomputed_positions[round_num + 1].tolist())
            
            # استفاده از ملاقات‌های از پیش محاسبه شده
            if self.precomputed_meetings and round_num < len(self.precomputed_meetings):
                meetings = self.precomputed_meetings[round_num]
            else:
                meetings = self.mobility_manager.get_meetings()
        else:
            # حرکت معمول (برای حالت fallback)
            mobility_seed = self.config.seed + self.run_id * 1000 + round_num * 100
            self.mobility_manager.move_nodes(mobility_seed)
            meetings = self.mobility_manager.get_meetings()
            self.results['positions_used'].append(self.mobility_manager.positions.tolist())
        
        # محاسبه ملاقات‌های هر گره
        meeting_counts = [0] * self.config.num_nodes
        for i, j, distance in meetings:
            meeting_counts[i] += 1
            meeting_counts[j] += 1
        
        # به‌روزرسانی گراف‌ها
        temp_graphs = [self.local_graphs[i].copy() for i in range(self.config.num_nodes)]
        
        for i, j, distance in meetings:
            w_i = self.graph_manager.calculate_node_weight(
                self.data_stats[i]['size'],
                self.accuracies[i],
                self.data_stats[i]['entropy']
            )
            w_j = self.graph_manager.calculate_node_weight(
                self.data_stats[j]['size'],
                self.accuracies[j],
                self.data_stats[j]['entropy']
            )
            
            dist = self.graph_manager.calculate_distance(
                self.graph_manager.extract_model_weights(self.models[i]),
                self.graph_manager.extract_model_weights(self.models[j]),
                self.config.distance_method
            )
            
            if j not in temp_graphs[i].nodes():
                temp_graphs[i].add_node(j, weight=w_j)
            if i not in temp_graphs[j].nodes():
                temp_graphs[j].add_node(i, weight=w_i)
            
            temp_graphs[i].add_edge(i, j, weight=dist)
            temp_graphs[j].add_edge(j, i, weight=dist)
        
        for i in range(self.config.num_nodes):
            received_graphs = []
            neighbors = list(temp_graphs[i].neighbors(i))
            for neighbor in neighbors:
                if neighbor != i:
                    received_graphs.append(temp_graphs[neighbor])
            
            self.local_graphs[i] = self.combine_graphs(temp_graphs[i], received_graphs)
        
        # انتخاب مجموعه برتری و ادغام
        data_sizes = [stats['size'] for stats in self.data_stats]
        node_weights = []
        dominating_set_sizes = []
        
        for i in range(self.config.num_nodes):
            models_weights = [self.graph_manager.extract_model_weights(model) for model in self.models]
            dominating_set = self.ds_selector.select_dominating_set(
                self.local_graphs[i], data_sizes, self.accuracies, models_weights
            )
            
            dominating_set_sizes.append(len(dominating_set))
            
            if not dominating_set:
                continue
            
            criteria_matrix = []
            for node in dominating_set:
                criteria_matrix.append([
                    self.accuracies[node],
                    data_sizes[node],
                    self.local_graphs[i].nodes[node]['weight']
                ])
            
            criteria_matrix = np.array(criteria_matrix)
            
            
            criteria_weights, cr = self.ahp_waspas.calculate_ahp_weights(criteria_comparison)
            criteria_types = ['max', 'max', 'max']
            final_weights = self.ahp_waspas.calculate_waspas_scores(criteria_matrix, criteria_weights, criteria_types)
            
            node_weights.append({
                'node_id': i,
                'dominating_set': [int(x) for x in dominating_set],
                'final_weights': [float(w) for w in final_weights.tolist()]
            })
            
            # ذخیره وزن‌های قبلی گره i
            old_weights = self.models[i].get_weights()
            
            # ادغام وزن‌ها از مجموعه برتری
            aggregated_weights = []
            num_layers = len(old_weights)
            
            for layer_idx in range(num_layers):
                layer_weights_sum = np.zeros_like(old_weights[layer_idx])
                
                for idx, node in enumerate(dominating_set):
                    node_layer_weights = self.models[node].get_weights()[layer_idx]
                    layer_weights_sum += final_weights[idx] * node_layer_weights
                
                aggregated_weights.append(layer_weights_sum)
            
            # ترکیب وزن‌های قدیمی و جدید با ضریب dfl_beta
            beta = self.config.dfl_beta
            combined_weights = []
            
            for old_layer, new_layer in zip(old_weights, aggregated_weights):
                combined_layer = (1 - beta) * old_layer + beta * new_layer
                combined_weights.append(combined_layer)
            
            # به‌روزرسانی مدل با وزن‌های ترکیبی
            self.models[i].set_weights(combined_weights)
        
        # محاسبه شباهت مدل‌ها
        model_similarities = self.graph_manager.calculate_model_similarity(self.models)
        
        # ذخیره نتایج
        self.results['accuracy'].append([float(x) for x in round_accuracies])
        self.results['f1_scores'].append([float(x) for x in round_f1_scores])
        self.results['loss'].append([float(x) for x in round_losses])
        self.results['meetings_per_node'].append([int(x) for x in meeting_counts])
        self.results['model_similarities'].append(model_similarities)
        self.results['node_weights'].append(node_weights)
        self.results['dominating_set_sizes'].append([int(x) for x in dominating_set_sizes])
        
        # ذخیره دقت و loss هر گره به صورت جداگانه
        individual_acc = {}
        individual_loss = {}
        for node_id, (acc, loss_val) in enumerate(zip(round_accuracies, round_losses)):
            individual_acc[f'node_{node_id}'] = float(acc)
            individual_loss[f'node_{node_id}'] = float(loss_val)
        
        self.results['individual_accuracies'].append(individual_acc)
        self.results['individual_losses'].append(individual_loss)
        
        # محاسبه میانگین‌ها
        mean_accuracy = float(np.mean(round_accuracies))
        mean_f1 = float(np.mean(round_f1_scores))
        mean_loss = float(np.mean(round_losses))
        mean_meetings = float(np.mean(meeting_counts))
        
        round_time = time.time() - round_start_time
        
        return mean_accuracy, mean_f1, mean_loss, mean_meetings, round_time
    
    def combine_graphs(self, graph1: nx.Graph, received_graphs: List[nx.Graph]) -> nx.Graph:
        """ترکیب گراف‌ها"""
        combined_graph = graph1.copy()
        
        for g in received_graphs:
            for node, data in g.nodes(data=True):
                if node in combined_graph.nodes():
                    combined_graph.nodes[node]['weight'] = (combined_graph.nodes[node]['weight'] + data['weight']) / 2
                else:
                    combined_graph.add_node(node, **data)
            
            for u, v, data in g.edges(data=True):
                if combined_graph.has_edge(u, v):
                    combined_graph[u][v]['weight'] = (combined_graph[u][v]['weight'] + data['weight']) / 2
                else:
                    combined_graph.add_edge(u, v, **data)
        
        return combined_graph
    
    def run(self):
        """اجرای کامل"""
        self.initialize()
        
        overall_accuracies = []
        overall_f1_scores = []
        overall_losses = []
        round_times = []
        
        for round_num in range(self.config.num_rounds):
            mean_acc, mean_f1, mean_loss, mean_meetings, round_time = self.run_round(round_num)
            overall_accuracies.append(mean_acc)
            overall_f1_scores.append(mean_f1)
            overall_losses.append(mean_loss)
            round_times.append(round_time)
            
            if (round_num + 1) % max(1, self.config.num_rounds // 10) == 0:
                print(f"  Run {self.run_id+1}, Round {round_num+1}: Accuracy={mean_acc:.4f}, Loss={mean_loss:.4f}, F1={mean_f1:.4f}")
        
        return overall_accuracies, overall_f1_scores, overall_losses, self.results, round_times

class FedAvgP2P_WithSharedMobility:
    """FedAvg P2P با مسیرهای تحرک مشترک"""
    def __init__(self, config: Config, run_id: int = 0,
                 precomputed_positions: List[np.ndarray] = None,
                 precomputed_meetings: List[List[Tuple]] = None):
        self.config = config
        self.run_id = run_id
        self.precomputed_positions = precomputed_positions
        self.precomputed_meetings = precomputed_meetings
        
        self.data_handler = DataHandler(config)
        self.model_handler = ModelHandler(config, self.data_handler.num_classes, self.data_handler.input_shape)
        self.mobility_manager = MobilityManager(config)
        self.graph_manager = GraphManager(config)  # اضافه شد برای محاسبه similarity
        
        self.models = []
        self.client_datasets = []
        self.data_stats = []
        self.accuracies = []
        
        self.results = {
            'accuracy': [],
            'f1_scores': [],
            'loss': [],
            'meetings_per_node': [],
            'model_similarities': [],  # اضافه شد
            'individual_accuracies': [],
            'individual_losses': [],  # اضافه شد
            'positions_used': []  # ذخیره موقعیت‌های استفاده شده
        }
    
    def initialize(self, seed_offset: int = 0):
        # استفاده از توزیع داده ثابت
        self.client_datasets, self.data_stats = self.data_handler.get_fixed_client_datasets(
            self.config.num_nodes
        )
        
        self.models = [self.model_handler.create_model() for _ in range(self.config.num_nodes)]
        
        # مقداردهی اولیه موقعیت‌ها با اولین موقعیت از پیش محاسبه شده
        if self.precomputed_positions:
            self.mobility_manager.positions = self.precomputed_positions[0].copy()
        else:
            mobility_seed = self.config.seed + self.run_id * 1000 + seed_offset
            self.mobility_manager.initialize_positions(self.config.num_nodes, mobility_seed)
        
        self.accuracies = [0.0] * self.config.num_nodes
    
    def train_single_node(self, node_id: int):
        """آموزش یک گره"""
        model = self.models[node_id]
        dataset = self.client_datasets[node_id]
        
        # آموزش مدل
        history = model.fit(
            dataset,
            epochs=self.config.local_epochs,
            verbose=0
        )
        
        # پیش‌بینی
        predictions = model.predict(self.data_handler.x_test, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)
        
        accuracy = float(accuracy_score(self.data_handler.y_test, pred_labels))
        f1 = float(f1_score(self.data_handler.y_test, pred_labels, average='macro'))
        loss = float(history.history['loss'][-1])
        
        self.accuracies[node_id] = accuracy
        
        return loss, accuracy, f1, history.history
    
    def local_train_sequential(self):
        """آموزش ترتیبی همه گره‌ها"""
        losses = []
        accuracies = []
        f1_scores = []
        histories = []
        
        # اجرای ترتیبی
        for i in range(self.config.num_nodes):
            loss, acc, f1, history = self.train_single_node(i)
            losses.append(loss)
            accuracies.append(acc)
            f1_scores.append(f1)
            histories.append(history)
        
        return losses, accuracies, f1_scores, histories
    
    def run_round(self, round_num: int):
        """اجرای یک دور"""
        round_start_time = time.time()
        
        round_losses, round_accuracies, round_f1_scores, round_histories = self.local_train_sequential()
        
        # استفاده از موقعیت‌های از پیش محاسبه شده
        if self.precomputed_positions and round_num < len(self.precomputed_positions) - 1:
            self.mobility_manager.positions = self.precomputed_positions[round_num + 1].copy()
            self.results['positions_used'].append(self.precomputed_positions[round_num + 1].tolist())
            
            # استفاده از ملاقات‌های از پیش محاسبه شده
            if self.precomputed_meetings and round_num < len(self.precomputed_meetings):
                meetings = self.precomputed_meetings[round_num]
            else:
                meetings = self.mobility_manager.get_meetings()
        else:
            # حرکت معمول (برای حالت fallback)
            mobility_seed = self.config.seed + self.run_id * 1000 + round_num * 100
            self.mobility_manager.move_nodes(mobility_seed)
            meetings = self.mobility_manager.get_meetings()
            self.results['positions_used'].append(self.mobility_manager.positions.tolist())
        
        # محاسبه ملاقات‌های هر گره و ذخیره اطلاعات
        meeting_counts = [0] * self.config.num_nodes
        meeting_partners = [[] for _ in range(self.config.num_nodes)]
        meeting_data = []  # برای ذخیره همه ملاقات‌ها
        
        for i, j, distance in meetings:
            meeting_counts[i] += 1
            meeting_counts[j] += 1
            meeting_partners[i].append(j)
            meeting_partners[j].append(i)
            meeting_data.append((i, j, distance))
        
        # تجمیع برای هر گره بر اساس همه گره‌هایی که با آنها ملاقات کرده است
        aggregated_models = []
        
        for node_id in range(self.config.num_nodes):
            # لیست همه گره‌هایی که با آنها ملاقات کرده (شامل خودش)
            all_partners = list(set([node_id] + meeting_partners[node_id]))
            
            if len(all_partners) > 1:
                # محاسبه وزن‌ها بر اساس حجم داده
                total_data_size = 0
                for partner in all_partners:
                    total_data_size += self.data_stats[partner]['size']
                
                # اگر حجم کل داده صفر باشد (که معمولاً نیست)، از وزن مساوی استفاده کن
                if total_data_size == 0:
                    weights = [1.0 / len(all_partners)] * len(all_partners)
                else:
                    weights = [self.data_stats[partner]['size'] / total_data_size for partner in all_partners]
                
                # تجمیع وزن‌های مدل
                aggregated_weights = []
                num_layers = len(self.models[node_id].get_weights())
                
                for layer_idx in range(num_layers):
                    layer_weights_sum = np.zeros_like(self.models[node_id].get_weights()[layer_idx])
                    
                    for idx, partner_id in enumerate(all_partners):
                        partner_layer_weights = self.models[partner_id].get_weights()[layer_idx]
                        layer_weights_sum += weights[idx] * partner_layer_weights
                    
                    aggregated_weights.append(layer_weights_sum)
                
                aggregated_models.append({
                    'node_id': node_id,
                    'weights': aggregated_weights
                })
            else:
                # اگر با هیچ گره‌ای ملاقات نکرده، مدلش بدون تغییر می‌ماند
                aggregated_models.append({
                    'node_id': node_id,
                    'weights': self.models[node_id].get_weights()
                })
        
        # به‌روزرسانی مدل‌ها پس از محاسبه همه تجمیع‌ها
        for agg_model in aggregated_models:
            self.models[agg_model['node_id']].set_weights(agg_model['weights'])
        
        # محاسبه شباهت مدل‌ها
        model_similarities = self.graph_manager.calculate_model_similarity(self.models)
        
        # ذخیره نتایج
        self.results['accuracy'].append([float(x) for x in round_accuracies])
        self.results['f1_scores'].append([float(x)for x in round_f1_scores])
        self.results['loss'].append([float(x) for x in round_losses])
        self.results['meetings_per_node'].append([int(x) for x in meeting_counts])
        self.results['model_similarities'].append(model_similarities)
        
        # ذخیره دقت و loss هر گره به صورت جداگانه
        individual_acc = {}
        individual_loss = {}
        for node_id, (acc, loss_val) in enumerate(zip(round_accuracies, round_losses)):
            individual_acc[f'node_{node_id}'] = float(acc)
            individual_loss[f'node_{node_id}'] = float(loss_val)
        
        self.results['individual_accuracies'].append(individual_acc)
        self.results['individual_losses'].append(individual_loss)
        
        # محاسبه میانگین‌ها
        mean_accuracy = float(np.mean(round_accuracies))
        mean_f1 = float(np.mean(round_f1_scores))
        mean_loss = float(np.mean(round_losses))
        mean_meetings = float(np.mean(meeting_counts))
        
        round_time = time.time() - round_start_time
        
        return mean_accuracy, mean_f1, mean_loss, mean_meetings, round_time
    
    def run(self):
        """اجرای کامل"""
        self.initialize()
        
        overall_accuracies = []
        overall_f1_scores = []
        overall_losses = []
        round_times = []
        
        for round_num in range(self.config.num_rounds):
            mean_acc, mean_f1, mean_loss, mean_meetings, round_time = self.run_round(round_num)
            overall_accuracies.append(mean_acc)
            overall_f1_scores.append(mean_f1)
            overall_losses.append(mean_loss)
            round_times.append(round_time)
            
            if (round_num + 1) % max(1, self.config.num_rounds // 10) == 0:
                print(f"  Run {self.run_id+1}, Round {round_num+1}: Accuracy={mean_acc:.4f}, Loss={mean_loss:.4f}, F1={mean_f1:.4f}")
        
        return overall_accuracies, overall_f1_scores, overall_losses, self.results, round_times

class WAFL_WithSharedMobility:
    """WAFL (Wireless Ad Hoc Federated Learning) با مسیرهای تحرک مشترک"""
    def __init__(self, config: Config, run_id: int = 0,
                 precomputed_positions: List[np.ndarray] = None,
                 precomputed_meetings: List[List[Tuple]] = None):
        self.config = config
        self.run_id = run_id
        self.precomputed_positions = precomputed_positions
        self.precomputed_meetings = precomputed_meetings
        
        self.data_handler = DataHandler(config)
        self.model_handler = ModelHandler(config, self.data_handler.num_classes, self.data_handler.input_shape)
        self.mobility_manager = MobilityManager(config)
        self.graph_manager = GraphManager(config)
        
        self.models = []
        self.client_datasets = []
        self.data_stats = []
        self.accuracies = []
        
        self.results = {
            'accuracy': [],
            'f1_scores': [],
            'loss': [],
            'meetings_per_node': [],
            'model_similarities': [],
            'individual_accuracies': [],
            'individual_losses': [],
            'positions_used': [],
            'aggregation_info': []  # اطلاعات تجمیع WAFL
        }
    
    def initialize(self, seed_offset: int = 0):
        # استفاده از توزیع داده ثابت
        self.client_datasets, self.data_stats = self.data_handler.get_fixed_client_datasets(
            self.config.num_nodes
        )
        
        self.models = [self.model_handler.create_model() for _ in range(self.config.num_nodes)]
        
        # مقداردهی اولیه موقعیت‌ها با اولین موقعیت از پیش محاسبه شده
        if self.precomputed_positions:
            self.mobility_manager.positions = self.precomputed_positions[0].copy()
        else:
            mobility_seed = self.config.seed + self.run_id * 1000 + seed_offset
            self.mobility_manager.initialize_positions(self.config.num_nodes, mobility_seed)
        
        self.accuracies = [0.0] * self.config.num_nodes
    
    def train_single_node(self, node_id: int):
        """آموزش یک گره"""
        model = self.models[node_id]
        dataset = self.client_datasets[node_id]
        
        # آموزش مدل
        history = model.fit(
            dataset,
            epochs=self.config.local_epochs,
            verbose=0
        )
        
        # پیش‌بینی
        predictions = model.predict(self.data_handler.x_test, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)
        
        accuracy = float(accuracy_score(self.data_handler.y_test, pred_labels))
        f1 = float(f1_score(self.data_handler.y_test, pred_labels, average='macro'))
        loss = float(history.history['loss'][-1])
        
        self.accuracies[node_id] = accuracy
        
        return loss, accuracy, f1, history.history
    
    def local_train_sequential(self):
        """آموزش ترتیبی همه گره‌ها"""
        losses = []
        accuracies = []
        f1_scores = []
        histories = []
        
        # اجرای ترتیبی
        for i in range(self.config.num_nodes):
            loss, acc, f1, history = self.train_single_node(i)
            losses.append(loss)
            accuracies.append(acc)
            f1_scores.append(f1)
            histories.append(history)
        
        return losses, accuracies, f1_scores, histories
    
    def run_round(self, round_num: int):
        """اجرای یک دور"""
        round_start_time = time.time()
        
        round_losses, round_accuracies, round_f1_scores, round_histories = self.local_train_sequential()
        
        # استفاده از موقعیت‌های از پیش محاسبه شده
        if self.precomputed_positions and round_num < len(self.precomputed_positions) - 1:
            self.mobility_manager.positions = self.precomputed_positions[round_num + 1].copy()
            self.results['positions_used'].append(self.precomputed_positions[round_num + 1].tolist())
            
            # استفاده از ملاقات‌های از پیش محاسبه شده
            if self.precomputed_meetings and round_num < len(self.precomputed_meetings):
                meetings = self.precomputed_meetings[round_num]
            else:
                meetings = self.mobility_manager.get_meetings()
        else:
            # حرکت معمول (برای حالت fallback)
            mobility_seed = self.config.seed + self.run_id * 1000 + round_num * 100
            self.mobility_manager.move_nodes(mobility_seed)
            meetings = self.mobility_manager.get_meetings()
            self.results['positions_used'].append(self.mobility_manager.positions.tolist())
        
        # محاسبه ملاقات‌های هر گره
        meeting_counts = [0] * self.config.num_nodes
        meeting_partners = [[] for _ in range(self.config.num_nodes)]
        
        for i, j, distance in meetings:
            meeting_counts[i] += 1
            meeting_counts[j] += 1
            meeting_partners[i].append(j)
            meeting_partners[j].append(i)
        
        # انجام WAFL aggregation برای هر گره
        aggregation_info = []
        
        for node_id in range(self.config.num_nodes):
            neighbors = meeting_partners[node_id]
            
            if len(neighbors) > 0:
                # ذخیره مدل فعلی
                current_weights = self.models[node_id].get_weights()
                
                # محاسبه aggregate بر اساس فرمول WAFL
                # θ_new = θ_old + λ * Σ(θ_neighbor - θ_old) / (n + 1)
                # که n تعداد همسایه‌ها است
                
                # جمع تفاوت‌ها
                total_diff = []
                for layer_idx in range(len(current_weights)):
                    total_diff.append(np.zeros_like(current_weights[layer_idx]))
                
                for neighbor_id in neighbors:
                    neighbor_weights = self.models[neighbor_id].get_weights()
                    for layer_idx in range(len(current_weights)):
                        total_diff[layer_idx] += (neighbor_weights[layer_idx] - current_weights[layer_idx])
                
                # اعمال فرمول WAFL
                new_weights = []
                n = len(neighbors)
                lambda_coeff = self.config.wafl_lambda
                
                for layer_idx in range(len(current_weights)):
                    # فرمول: θ_new = θ_old + λ * Σ(θ_neighbor - θ_old) / (n + 1)
                    if n > 0:
                        layer_new = current_weights[layer_idx] + lambda_coeff * total_diff[layer_idx] / (n + 1)
                    else:
                        layer_new = current_weights[layer_idx]
                    new_weights.append(layer_new)
                
                # به‌روزرسانی مدل
                self.models[node_id].set_weights(new_weights)
                
                aggregation_info.append({
                    'node_id': node_id,
                    'num_neighbors': n,
                    'lambda_used': lambda_coeff
                })
        
        # محاسبه شباهت مدل‌ها
        model_similarities = self.graph_manager.calculate_model_similarity(self.models)
        
        # ذخیره نتایج
        self.results['accuracy'].append([float(x) for x in round_accuracies])
        self.results['f1_scores'].append([float(x) for x in round_f1_scores])
        self.results['loss'].append([float(x) for x in round_losses])
        self.results['meetings_per_node'].append([int(x) for x in meeting_counts])
        self.results['model_similarities'].append(model_similarities)
        self.results['aggregation_info'].append(aggregation_info)
        
        # ذخیره دقت و loss هر گره به صورت جداگانه
        individual_acc = {}
        individual_loss = {}
        for node_id, (acc, loss_val) in enumerate(zip(round_accuracies, round_losses)):
            individual_acc[f'node_{node_id}'] = float(acc)
            individual_loss[f'node_{node_id}'] = float(loss_val)
        
        self.results['individual_accuracies'].append(individual_acc)
        self.results['individual_losses'].append(individual_loss)
        
        # محاسبه میانگین‌ها
        mean_accuracy = float(np.mean(round_accuracies))
        mean_f1 = float(np.mean(round_f1_scores))
        mean_loss = float(np.mean(round_losses))
        mean_meetings = float(np.mean(meeting_counts))
        
        round_time = time.time() - round_start_time
        
        return mean_accuracy, mean_f1, mean_loss, mean_meetings, round_time
    
    def run(self):
        """اجرای کامل"""
        self.initialize()
        
        overall_accuracies = []
        overall_f1_scores = []
        overall_losses = []
        round_times = []
        
        for round_num in range(self.config.num_rounds):
            mean_acc, mean_f1, mean_loss, mean_meetings, round_time = self.run_round(round_num)
            overall_accuracies.append(mean_acc)
            overall_f1_scores.append(mean_f1)
            overall_losses.append(mean_loss)
            round_times.append(round_time)
            
            if (round_num + 1) % max(1, self.config.num_rounds // 10) == 0:
                print(f"  Run {self.run_id+1}, Round {round_num+1}: Accuracy={mean_acc:.4f}, Loss={mean_loss:.4f}, F1={mean_f1:.4f}")
        
        return overall_accuracies, overall_f1_scores, overall_losses, self.results, round_times

class MultiRunExperimentWithSharedMobility:
    """اجرای آزمایش‌های چندگانه با تحرک مشترک"""
    def __init__(self, config: Config):
        self.config = config
        self.save_dir = config.create_save_dir()
        
        # مولد مسیرهای تحرک
        self.trajectory_generator = MobilityTrajectoryGenerator(config)
        
        self.all_dfl_results = []
        self.all_fedavg_results = []
        self.all_wafl_results = []
        
        # ذخیره دقت‌های هر گره در هر راند برای هر اجرا
        self.dfl_individual_accuracies = []  # ساختار: [runs][rounds][nodes]
        self.fedavg_individual_accuracies = []
        self.wafl_individual_accuracies = []
        
        # ذخیره loss هر گره در هر راند برای هر اجرا
        self.dfl_individual_losses = []  # ساختار: [runs][rounds][nodes]
        self.fedavg_individual_losses = []
        self.wafl_individual_losses = []
        
        # ذخیره similarity هر راند برای هر اجرا
        self.dfl_similarities = []  # ساختار: [runs][rounds][metrics]
        self.fedavg_similarities = []
        self.wafl_similarities = []
        
        self.dfl_accuracies_over_runs = []  # میانگین دقت در هر راند برای هر اجرا
        self.dfl_f1_over_runs = []
        self.dfl_losses_over_runs = []
        self.fedavg_accuracies_over_runs = []
        self.fedavg_f1_over_runs = []
        self.fedavg_losses_over_runs = []
        self.wafl_accuracies_over_runs = []
        self.wafl_f1_over_runs = []
        self.wafl_losses_over_runs = []
        
        self.dfl_round_times = []
        self.fedavg_round_times = []
        self.wafl_round_times = []
        
    def run(self):
        """اجرای N بار شبیه‌سازی با تحرک مشترک"""
        print(f"\n{'='*70}")
        print(f"STARTING MULTI-RUN EXPERIMENT WITH SHARED MOBILITY")
        print(f"Number of runs: {self.config.num_runs}")
        print(f"Number of nodes: {self.config.num_nodes}")
        print(f"Number of rounds: {self.config.num_rounds}")
        print(f"Dataset: {self.config.dataset}")
        print(f"Mobility model: {self.config.mobility_model}")
        print(f"Results will be saved in: {self.save_dir}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        for run_id in range(self.config.num_runs):
            print(f"\n{'='*50}")
            print(f"Run {run_id + 1}/{self.config.num_runs}")
            print(f"{'='*50}")
            
            # 1. تولید مسیرهای تحرک مشترک برای این اجرا
            print("Generating shared mobility trajectories...")
            positions_list, meetings_list = self.trajectory_generator.generate_trajectory(run_id)
            
            # ذخیره مسیرهای تحرک
            mobility_data = {
                'run_id': run_id + 1,
                'positions': [pos.tolist() for pos in positions_list],
                'meetings': meetings_list
            }
            
            run_dir = os.path.join(self.save_dir, f"run_{run_id+1}")
            os.makedirs(run_dir, exist_ok=True)
            
            with open(os.path.join(run_dir, 'mobility_trajectories.json'), 'w') as f:
                json.dump(mobility_data, f, cls=JSONEncoder, indent=4)
            
            # 2. اجرای DFL-DGA با مسیرهای مشترک
            print("Running DFL-DGA with shared mobility...")
            dfl_start = time.time()
            dfl_dga = DFL_DGA_WithSharedMobility(
                self.config, run_id, 
                precomputed_positions=positions_list,
                precomputed_meetings=meetings_list
            )
            dfl_accuracies, dfl_f1_scores, dfl_losses, dfl_results, dfl_times = dfl_dga.run()
            dfl_time = time.time() - dfl_start
            print(f"  DFL-DGA completed in {dfl_time:.2f} seconds")
            
            self.dfl_accuracies_over_runs.append(dfl_accuracies)
            self.dfl_f1_over_runs.append(dfl_f1_scores)
            self.dfl_losses_over_runs.append(dfl_losses)
            self.dfl_round_times.append(dfl_times)
            self.all_dfl_results.append(dfl_results)
            
            # استخراج دقت‌ها و lossهای فردی هر گره
            dfl_individual_acc = []
            dfl_individual_loss = []
            for round_data in dfl_results['accuracy']:
                dfl_individual_acc.append([float(x) for x in round_data])
            for round_data in dfl_results['loss']:
                dfl_individual_loss.append([float(x) for x in round_data])
            
            self.dfl_individual_accuracies.append(dfl_individual_acc)
            self.dfl_individual_losses.append(dfl_individual_loss)
            
            # استخراج similarity
            dfl_sim = []
            for sim_data in dfl_results['model_similarities']:
                dfl_sim.append({
                    'mean': sim_data['mean'],
                    'std': sim_data['std'],
                    'min': sim_data.get('min', 0),
                    'max': sim_data.get('max', 0)
                })
            self.dfl_similarities.append(dfl_sim)
            
            # 3. اجرای FedAvg P2P با همان مسیرهای مشترک
            print("\nRunning FedAvg P2P with shared mobility...")
            fedavg_start = time.time()
            fedavg = FedAvgP2P_WithSharedMobility(
                self.config, run_id,
                precomputed_positions=positions_list,
                precomputed_meetings=meetings_list
            )
            fedavg_accuracies, fedavg_f1_scores, fedavg_losses, fedavg_results, fedavg_times = fedavg.run()
            fedavg_time = time.time() - fedavg_start
            print(f"  FedAvg P2P completed in {fedavg_time:.2f} seconds")
            
            self.fedavg_accuracies_over_runs.append(fedavg_accuracies)
            self.fedavg_f1_over_runs.append(fedavg_f1_scores)
            self.fedavg_losses_over_runs.append(fedavg_losses)
            self.fedavg_round_times.append(fedavg_times)
            self.all_fedavg_results.append(fedavg_results)
            
            # استخراج دقت‌ها و lossهای فردی هر گره
            fedavg_individual_acc = []
            fedavg_individual_loss = []
            for round_data in fedavg_results['accuracy']:
                fedavg_individual_acc.append([float(x) for x in round_data])
            for round_data in fedavg_results['loss']:
                fedavg_individual_loss.append([float(x) for x in round_data])
            
            self.fedavg_individual_accuracies.append(fedavg_individual_acc)
            self.fedavg_individual_losses.append(fedavg_individual_loss)
            
            # استخراج similarity
            fedavg_sim = []
            for sim_data in fedavg_results['model_similarities']:
                fedavg_sim.append({
                    'mean': sim_data['mean'],
                    'std': sim_data['std'],
                    'min': sim_data.get('min', 0),
                    'max': sim_data.get('max', 0)
                })
            self.fedavg_similarities.append(fedavg_sim)
            
            # 4. اجرای WAFL با همان مسیرهای مشترک
            print("\nRunning WAFL with shared mobility...")
            wafl_start = time.time()
            wafl = WAFL_WithSharedMobility(
                self.config, run_id,
                precomputed_positions=positions_list,
                precomputed_meetings=meetings_list
            )
            wafl_accuracies, wafl_f1_scores, wafl_losses, wafl_results, wafl_times = wafl.run()
            wafl_time = time.time() - wafl_start
            print(f"  WAFL completed in {wafl_time:.2f} seconds")
            
            self.wafl_accuracies_over_runs.append(wafl_accuracies)
            self.wafl_f1_over_runs.append(wafl_f1_scores)
            self.wafl_losses_over_runs.append(wafl_losses)
            self.wafl_round_times.append(wafl_times)
            self.all_wafl_results.append(wafl_results)
            
            # استخراج دقت‌ها و lossهای فردی هر گره
            wafl_individual_acc = []
            wafl_individual_loss = []
            for round_data in wafl_results['accuracy']:
                wafl_individual_acc.append([float(x) for x in round_data])
            for round_data in wafl_results['loss']:
                wafl_individual_loss.append([float(x) for x in round_data])
            
            self.wafl_individual_accuracies.append(wafl_individual_acc)
            self.wafl_individual_losses.append(wafl_individual_loss)
            
            # استخراج similarity
            wafl_sim = []
            for sim_data in wafl_results['model_similarities']:
                wafl_sim.append({
                    'mean': sim_data['mean'],
                    'std': sim_data['std'],
                    'min': sim_data.get('min', 0),
                    'max': sim_data.get('max', 0)
                })
            self.wafl_similarities.append(wafl_sim)
            
            # ذخیره نتایج این اجرا
            self.save_single_run_results(run_id, 
                                        dfl_accuracies, dfl_f1_scores, dfl_losses,
                                        fedavg_accuracies, fedavg_f1_scores, fedavg_losses,
                                        wafl_accuracies, wafl_f1_scores, wafl_losses,
                                        dfl_results, fedavg_results, wafl_results, mobility_data)
        
        total_time = time.time() - start_time
        print(f"\nTotal experiment time: {total_time:.2f} seconds")
        
        # محاسبه آمار و تست‌های آماری
        self.calculate_detailed_statistics()
        
        # رسم نمودارهای آماری
        self.plot_statistical_tests()
        
        # رسم نمودارهای دقت، loss و similarity
        self.plot_accuracy_loss_similarity()
        
        # ذخیره نتایج ناشی
        self.save_final_results()
        
        print(f"\n{'='*70}")
        print("MULTI-RUN EXPERIMENT WITH SHARED MOBILITY COMPLETED SUCCESSFULLY!")
        print(f"All results saved in: {self.save_dir}")
        print(f"{'='*70}")
    
    def save_single_run_results(self, run_id: int, 
                               dfl_accuracies: List[float], dfl_f1_scores: List[float], dfl_losses: List[float],
                               fedavg_accuracies: List[float], fedavg_f1_scores: List[float], fedavg_losses: List[float],
                               wafl_accuracies: List[float], wafl_f1_scores: List[float], wafl_losses: List[float],
                               dfl_results: Dict, fedavg_results: Dict, wafl_results: Dict, mobility_data: Dict):
        """ذخیره نتایج یک اجرا"""
        run_dir = os.path.join(self.save_dir, f"run_{run_id+1}")
        os.makedirs(run_dir, exist_ok=True)
        
        run_results = {
            'run_id': run_id + 1,
            'dfl_dga': {
                'accuracies': [float(x) for x in dfl_accuracies],
                'f1_scores': [float(x) for x in dfl_f1_scores],
                'losses': [float(x) for x in dfl_losses],
                'final_accuracy': float(dfl_accuracies[-1]) if dfl_accuracies else 0,
                'final_f1': float(dfl_f1_scores[-1]) if dfl_f1_scores else 0,
                'final_loss': float(dfl_losses[-1]) if dfl_losses else 0,
                'individual_accuracies': dfl_results['individual_accuracies'],
                'individual_losses': dfl_results.get('individual_losses', []),
                'model_similarities': dfl_results.get('model_similarities', []),
                'positions_used': dfl_results.get('positions_used', []),
                'dominating_set_sizes': dfl_results.get('dominating_set_sizes', [])
            },
            'fedavg_p2p': {
                'accuracies': [float(x) for x in fedavg_accuracies],
                'f1_scores': [float(x) for x in fedavg_f1_scores],
                'losses': [float(x) for x in fedavg_losses],
                'final_accuracy': float(fedavg_accuracies[-1]) if fedavg_accuracies else 0,
                'final_f1': float(fedavg_f1_scores[-1]) if fedavg_f1_scores else 0,
                'final_loss': float(fedavg_losses[-1]) if fedavg_losses else 0,
                'individual_accuracies': fedavg_results['individual_accuracies'],
                'individual_losses': fedavg_results.get('individual_losses', []),
                'model_similarities': fedavg_results.get('model_similarities', []),
                'positions_used': fedavg_results.get('positions_used', [])
            },
            'wafl': {
                'accuracies': [float(x) for x in wafl_accuracies],
                'f1_scores': [float(x) for x in wafl_f1_scores],
                'losses': [float(x) for x in wafl_losses],
                'final_accuracy': float(wafl_accuracies[-1]) if wafl_accuracies else 0,
                'final_f1': float(wafl_f1_scores[-1]) if wafl_f1_scores else 0,
                'final_loss': float(wafl_losses[-1]) if wafl_losses else 0,
                'individual_accuracies': wafl_results['individual_accuracies'],
                'individual_losses': wafl_results.get('individual_losses', []),
                'model_similarities': wafl_results.get('model_similarities', []),
                'positions_used': wafl_results.get('positions_used', []),
                'aggregation_info': wafl_results.get('aggregation_info', [])
            },
            'mobility_data': mobility_data
        }
        
        with open(os.path.join(run_dir, 'results.json'), 'w') as f:
            json.dump(run_results, f, cls=JSONEncoder, indent=4)
        
        # ذخیره دقت و loss هر گره به صورت CSV
        self.save_individual_metrics_csv(run_id, dfl_results, fedavg_results, wafl_results, run_dir)
    
    def save_individual_metrics_csv(self, run_id: int, dfl_results: Dict, fedavg_results: Dict, wafl_results: Dict, run_dir: str):
        """ذخیره دقت و loss هر گره به صورت CSV"""
        rounds = list(range(1, self.config.num_rounds + 1))
        
        all_data = []
        
        # برای DFL-DGA
        for round_idx, (acc_round, loss_round) in enumerate(zip(dfl_results['accuracy'], dfl_results['loss'])):
            for node_idx, (accuracy, loss) in enumerate(zip(acc_round, loss_round)):
                all_data.append({
                    'run': run_id + 1,
                    'round': round_idx + 1,
                    'node': node_idx + 1,
                    'algorithm': 'DFL-DGA',
                    'accuracy': float(accuracy),
                    'loss': float(loss)
                })
        
        # برای FedAvg
        for round_idx, (acc_round, loss_round) in enumerate(zip(fedavg_results['accuracy'], fedavg_results['loss'])):
            for node_idx, (accuracy, loss) in enumerate(zip(acc_round, loss_round)):
                all_data.append({
                    'run': run_id + 1,
                    'round': round_idx + 1,
                    'node': node_idx + 1,
                    'algorithm': 'FedAvg',
                    'accuracy': float(accuracy),
                    'loss': float(loss)
                })
        
        # برای WAFL
        for round_idx, (acc_round, loss_round) in enumerate(zip(wafl_results['accuracy'], wafl_results['loss'])):
            for node_idx, (accuracy, loss) in enumerate(zip(acc_round, loss_round)):
                all_data.append({
                    'run': run_id + 1,
                    'round': round_idx + 1,
                    'node': node_idx + 1,
                    'algorithm': 'WAFL',
                    'accuracy': float(accuracy),
                    'loss': float(loss)
                })
        
        df = pd.DataFrame(all_data)
        df.to_csv(os.path.join(run_dir, 'individual_metrics.csv'), index=False)
        
        # ذخیره similarity
        similarity_data = []
        
        # DFL-DGA similarity
        for round_idx, sim_data in enumerate(dfl_results.get('model_similarities', [])):
            similarity_data.append({
                'run': run_id + 1,
                'round': round_idx + 1,
                'algorithm': 'DFL-DGA',
                'similarity_mean': sim_data.get('mean', 0),
                'similarity_std': sim_data.get('std', 0),
                'similarity_min': sim_data.get('min', 0),
                'similarity_max': sim_data.get('max', 0)
            })
        
        # FedAvg similarity
        for round_idx, sim_data in enumerate(fedavg_results.get('model_similarities', [])):
            similarity_data.append({
                'run': run_id + 1,
                'round': round_idx + 1,
                'algorithm': 'FedAvg',
                'similarity_mean': sim_data.get('mean', 0),
                'similarity_std': sim_data.get('std', 0),
                'similarity_min': sim_data.get('min', 0),
                'similarity_max': sim_data.get('max', 0)
            })
        
        # WAFL similarity
        for round_idx, sim_data in enumerate(wafl_results.get('model_similarities', [])):
            similarity_data.append({
                'run': run_id + 1,
                'round': round_idx + 1,
                'algorithm': 'WAFL',
                'similarity_mean': sim_data.get('mean', 0),
                'similarity_std': sim_data.get('std', 0),
                'similarity_min': sim_data.get('min', 0),
                'similarity_max': sim_data.get('max', 0)
            })
        
        if similarity_data:
            df_sim = pd.DataFrame(similarity_data)
            df_sim.to_csv(os.path.join(run_dir, 'model_similarities.csv'), index=False)
        
        # ذخیره اندازه‌های مجموعه برتری
        if 'dominating_set_sizes' in dfl_results:
            dominating_data = []
            for round_idx, round_sizes in enumerate(dfl_results['dominating_set_sizes']):
                for node_idx, size in enumerate(round_sizes):
                    dominating_data.append({
                        'run': run_id + 1,
                        'round': round_idx + 1,
                        'node': node_idx + 1,
                        'dominating_set_size': int(size)
                    })
            
            df_dominating = pd.DataFrame(dominating_data)
            df_dominating.to_csv(os.path.join(run_dir, 'dominating_set_sizes.csv'), index=False)
        
        # ذخیره اطلاعات WAFL
        if 'aggregation_info' in wafl_results:
            wafl_info_data = []
            for round_idx, round_info in enumerate(wafl_results['aggregation_info']):
                for node_info in round_info:
                    wafl_info_data.append({
                        'run': run_id + 1,
                        'round': round_idx + 1,
                        'node': node_info['node_id'] + 1,
                        'num_neighbors': node_info['num_neighbors'],
                        'lambda_used': node_info['lambda_used']
                    })
            
            df_wafl = pd.DataFrame(wafl_info_data)
            df_wafl.to_csv(os.path.join(run_dir, 'wafl_info.csv'), index=False)
    
    def calculate_detailed_statistics(self):
        """محاسبه آمار تفصیلی و تست‌های آماری"""
        print("\n" + "="*70)
        print("CALCULATING DETAILED STATISTICS AND STATISTICAL TESTS")
        print("="*70)
        
        # تبدیل به آرایه‌های numpy برای محاسبات
        self.dfl_individual_array = np.array(self.dfl_individual_accuracies)  # شکل: (runs, rounds, nodes)
        self.fedavg_individual_array = np.array(self.fedavg_individual_accuracies)
        self.wafl_individual_array = np.array(self.wafl_individual_accuracies)
        
        self.dfl_loss_array = np.array(self.dfl_individual_losses)
        self.fedavg_loss_array = np.array(self.fedavg_individual_losses)
        self.wafl_loss_array = np.array(self.wafl_individual_losses)
        
        # 1. میانگین و انحراف معیار برای هر گره در هر راند
        self.dfl_node_round_mean = np.mean(self.dfl_individual_array, axis=0)  # (rounds, nodes)
        self.dfl_node_round_std = np.std(self.dfl_individual_array, axis=0)
        
        self.fedavg_node_round_mean = np.mean(self.fedavg_individual_array, axis=0)
        self.fedavg_node_round_std = np.std(self.fedavg_individual_array, axis=0)
        
        self.wafl_node_round_mean = np.mean(self.wafl_individual_array, axis=0)
        self.wafl_node_round_std = np.std(self.wafl_individual_array, axis=0)
        
        # 2. میانگین و انحراف معیار کلی در هر راند
        self.dfl_round_mean = np.mean(self.dfl_individual_array, axis=(0, 2))  # (rounds,)
        self.dfl_round_std = np.std(self.dfl_individual_array, axis=(0, 2))
        
        self.fedavg_round_mean = np.mean(self.fedavg_individual_array, axis=(0, 2))
        self.fedavg_round_std = np.std(self.fedavg_individual_array, axis=(0, 2))
        
        self.wafl_round_mean = np.mean(self.wafl_individual_array, axis=(0, 2))
        self.wafl_round_std = np.std(self.wafl_individual_array, axis=(0, 2))
        
        # 3. میانگین و انحراف معیار loss در هر راند
        self.dfl_loss_round_mean = np.mean(self.dfl_loss_array, axis=(0, 2))
        self.dfl_loss_round_std = np.std(self.dfl_loss_array, axis=(0, 2))
        
        self.fedavg_loss_round_mean = np.mean(self.fedavg_loss_array, axis=(0, 2))
        self.fedavg_loss_round_std = np.std(self.fedavg_loss_array, axis=(0, 2))
        
        self.wafl_loss_round_mean = np.mean(self.wafl_loss_array, axis=(0, 2))
        self.wafl_loss_round_std = np.std(self.wafl_loss_array, axis=(0, 2))
        
        # 4. محاسبه آمار similarity
        self.calculate_similarity_statistics()
        
        # 5. محاسبه آمار برای اندازه مجموعه برتری‌ها
        self.calculate_dominating_set_statistics()
        
        # 6. تست‌های آماری
        self.perform_statistical_tests()
        
        print("\nDetailed statistics calculated successfully!")
    
    def calculate_similarity_statistics(self):
        """محاسبه آمار مربوط به similarity مدل‌ها"""
        # تبدیل similarity به آرایه‌های numpy
        dfl_sim_means = []
        fedavg_sim_means = []
        wafl_sim_means = []
        
        for run_sims in self.dfl_similarities:
            run_means = [s['mean'] for s in run_sims]
            dfl_sim_means.append(run_means)
        
        for run_sims in self.fedavg_similarities:
            run_means = [s['mean'] for s in run_sims]
            fedavg_sim_means.append(run_means)
        
        for run_sims in self.wafl_similarities:
            run_means = [s['mean'] for s in run_sims]
            wafl_sim_means.append(run_means)
        
        if dfl_sim_means:
            self.dfl_sim_array = np.array(dfl_sim_means)
            self.dfl_sim_mean = np.mean(self.dfl_sim_array, axis=0)
            self.dfl_sim_std = np.std(self.dfl_sim_array, axis=0)
        
        if fedavg_sim_means:
            self.fedavg_sim_array = np.array(fedavg_sim_means)
            self.fedavg_sim_mean = np.mean(self.fedavg_sim_array, axis=0)
            self.fedavg_sim_std = np.std(self.fedavg_sim_array, axis=0)
        
        if wafl_sim_means:
            self.wafl_sim_array = np.array(wafl_sim_means)
            self.wafl_sim_mean = np.mean(self.wafl_sim_array, axis=0)
            self.wafl_sim_std = np.std(self.wafl_sim_array, axis=0)
    
    def calculate_dominating_set_statistics(self):
        """محاسبه آمار مربوط به اندازه مجموعه برتری‌ها"""
        if not self.all_dfl_results:
            return
        
        dominating_set_data = []
        for run_result in self.all_dfl_results:
            if 'dominating_set_sizes' in run_result:
                # میانگین اندازه مجموعه برتری در هر دور برای این اجرا
                run_dominating_means = []
                for round_data in run_result['dominating_set_sizes']:
                    if len(round_data) > 0:
                        run_dominating_means.append(np.mean(round_data))
                    else:
                        run_dominating_means.append(0)
                dominating_set_data.append(run_dominating_means)
        
        if dominating_set_data:
            self.dominating_set_array = np.array(dominating_set_data)  # (runs, rounds)
            self.dominating_round_mean = np.mean(self.dominating_set_array, axis=0)
            self.dominating_round_std = np.std(self.dominating_set_array, axis=0)
    
    def perform_statistical_tests(self):
        """انجام تست‌های آماری مختلف"""
        self.statistical_results = {
            'per_round_tests': [],
            'final_round_tests': {},
            'effect_sizes': [],
            'confidence_intervals': [],
            'pairwise_comparisons': []
        }
        
        # تست برای هر راند
        for round_idx in range(self.config.num_rounds):
            dfl_round_data = self.dfl_individual_array[:, round_idx, :].flatten()
            fedavg_round_data = self.fedavg_individual_array[:, round_idx, :].flatten()
            wafl_round_data = self.wafl_individual_array[:, round_idx, :].flatten()
            
            # حذف مقادیر NaN
            dfl_valid = dfl_round_data[~np.isnan(dfl_round_data)]
            fedavg_valid = fedavg_round_data[~np.isnan(fedavg_round_data)]
            wafl_valid = wafl_round_data[~np.isnan(wafl_round_data)]
            
            round_results = {
                'round': int(round_idx + 1),
                'descriptive_stats': {
                    'dfl_mean': float(np.mean(dfl_valid)) if len(dfl_valid) > 0 else 0,
                    'dfl_std': float(np.std(dfl_valid)) if len(dfl_valid) > 0 else 0,
                    'fedavg_mean': float(np.mean(fedavg_valid)) if len(fedavg_valid) > 0 else 0,
                    'fedavg_std': float(np.std(fedavg_valid)) if len(fedavg_valid) > 0 else 0,
                    'wafl_mean': float(np.mean(wafl_valid)) if len(wafl_valid) > 0 else 0,
                    'wafl_std': float(np.std(wafl_valid)) if len(wafl_valid) > 0 else 0
                }
            }
            
            # مقایسه‌های زوجی
            comparisons = []
            
            # DFL-DGA vs FedAvg
            if len(dfl_valid) > 1 and len(fedavg_valid) > 1:
                t_stat_dfl_fedavg, p_value_dfl_fedavg = ttest_rel(dfl_valid, fedavg_valid)
                n1, n2 = len(dfl_valid), len(fedavg_valid)
                pooled_std = np.sqrt(((n1-1)*np.var(dfl_valid) + (n2-1)*np.var(fedavg_valid)) / (n1+n2-2))
                cohens_d_dfl_fedavg = (np.mean(dfl_valid) - np.mean(fedavg_valid)) / pooled_std if pooled_std != 0 else 0
                
                comparisons.append({
                    'comparison': 'DFL-DGA vs FedAvg',
                    't_statistic': float(t_stat_dfl_fedavg),
                    'p_value': float(p_value_dfl_fedavg),
                    'cohens_d': float(cohens_d_dfl_fedavg)
                })
            
            # DFL-DGA vs WAFL
            if len(dfl_valid) > 1 and len(wafl_valid) > 1:
                t_stat_dfl_wafl, p_value_dfl_wafl = ttest_rel(dfl_valid, wafl_valid)
                n1, n2 = len(dfl_valid), len(wafl_valid)
                pooled_std = np.sqrt(((n1-1)*np.var(dfl_valid) + (n2-1)*np.var(wafl_valid)) / (n1+n2-2))
                cohens_d_dfl_wafl = (np.mean(dfl_valid) - np.mean(wafl_valid)) / pooled_std if pooled_std != 0 else 0
                
                comparisons.append({
                    'comparison': 'DFL-DGA vs WAFL',
                    't_statistic': float(t_stat_dfl_wafl),
                    'p_value': float(p_value_dfl_wafl),
                    'cohens_d': float(cohens_d_dfl_wafl)
                })
            
            # FedAvg vs WAFL
            if len(fedavg_valid) > 1 and len(wafl_valid) > 1:
                t_stat_fedavg_wafl, p_value_fedavg_wafl = ttest_rel(fedavg_valid, wafl_valid)
                n1, n2 = len(fedavg_valid), len(wafl_valid)
                pooled_std = np.sqrt(((n1-1)*np.var(fedavg_valid) + (n2-1)*np.var(wafl_valid)) / (n1+n2-2))
                cohens_d_fedavg_wafl = (np.mean(fedavg_valid) - np.mean(wafl_valid)) / pooled_std if pooled_std != 0 else 0
                
                comparisons.append({
                    'comparison': 'FedAvg vs WAFL',
                    't_statistic': float(t_stat_fedavg_wafl),
                    'p_value': float(p_value_fedavg_wafl),
                    'cohens_d': float(cohens_d_fedavg_wafl)
                })
            
            round_results['pairwise_comparisons'] = comparisons
            self.statistical_results['per_round_tests'].append(round_results)
        
        # تست برای دور نهایی
        final_round_idx = self.config.num_rounds - 1
        if len(self.statistical_results['per_round_tests']) > final_round_idx:
            self.statistical_results['final_round_tests'] = self.statistical_results['per_round_tests'][final_round_idx]
    
    def interpret_cohens_d(self, d: float) -> str:
        """تفسیر اندازه اثر Cohen's d"""
        d = float(d)
        if abs(d) < 0.2:
            return "Negligible"
        elif abs(d) < 0.5:
            return "Small"
        elif abs(d) < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def plot_statistical_tests(self):
        """رسم نمودارهای تست‌های آماری"""
        print("\nPlotting statistical test results...")
        
        rounds = list(range(1, self.config.num_rounds + 1))
        
        # 1. نمودار p-value در هر راند برای مقایسه‌های مختلف
        plt.figure(figsize=(14, 8))
        
        # استخراج p-value برای هر مقایسه
        p_values_dfl_fedavg = []
        p_values_dfl_wafl = []
        p_values_fedavg_wafl = []
        
        for test in self.statistical_results['per_round_tests']:
            for comparison in test['pairwise_comparisons']:
                if comparison['comparison'] == 'DFL-DGA vs FedAvg':
                    p_values_dfl_fedavg.append(comparison['p_value'])
                elif comparison['comparison'] == 'DFL-DGA vs WAFL':
                    p_values_dfl_wafl.append(comparison['p_value'])
                elif comparison['comparison'] == 'FedAvg vs WAFL':
                    p_values_fedavg_wafl.append(comparison['p_value'])
        
        plt.plot(rounds, p_values_dfl_fedavg, 'b-', linewidth=2, marker='o', label='DFL-DGA vs FedAvg')
        plt.plot(rounds, p_values_dfl_wafl, 'r-', linewidth=2, marker='s', label='DFL-DGA vs WAFL')
        plt.plot(rounds, p_values_fedavg_wafl, 'g-', linewidth=2, marker='^', label='FedAvg vs WAFL')
        
        plt.axhline(y=0.05, color='k', linestyle='--', alpha=0.7, label='α = 0.05')
        plt.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='α = 0.01')
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('p-value', fontsize=12)
        plt.title('Pairwise Statistical Significance Across Rounds', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'pairwise_p_values_across_rounds.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. نمودار اندازه اثر (Cohen's d) برای مقایسه‌ها
        plt.figure(figsize=(14, 8))
        
        effect_sizes_dfl_fedavg = []
        effect_sizes_dfl_wafl = []
        effect_sizes_fedavg_wafl = []
        
        for test in self.statistical_results['per_round_tests']:
            for comparison in test['pairwise_comparisons']:
                if comparison['comparison'] == 'DFL-DGA vs FedAvg':
                    effect_sizes_dfl_fedavg.append(comparison['cohens_d'])
                elif comparison['comparison'] == 'DFL-DGA vs WAFL':
                    effect_sizes_dfl_wafl.append(comparison['cohens_d'])
                elif comparison['comparison'] == 'FedAvg vs WAFL':
                    effect_sizes_fedavg_wafl.append(comparison['cohens_d'])
        
        plt.plot(rounds, effect_sizes_dfl_fedavg, 'b-', linewidth=2, marker='o', label='DFL-DGA vs FedAvg')
        plt.plot(rounds, effect_sizes_dfl_wafl, 'r-', linewidth=2, marker='s', label='DFL-DGA vs WAFL')
        plt.plot(rounds, effect_sizes_fedavg_wafl, 'g-', linewidth=2, marker='^', label='FedAvg vs WAFL')
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # مناطق تفسیر اندازه اثر
        plt.axhspan(-0.2, 0.2, alpha=0.2, color='gray', label='Negligible')
        plt.axhspan(-0.5, -0.2, alpha=0.2, color='yellow')
        plt.axhspan(0.2, 0.5, alpha=0.2, color='yellow', label='Small')
        plt.axhspan(-0.8, -0.5, alpha=0.2, color='orange')
        plt.axhspan(0.5, 0.8, alpha=0.2, color='orange', label='Medium')
        plt.axhspan(-2, -0.8, alpha=0.2, color='red')
        plt.axhspan(0.8, 2, alpha=0.2, color='red', label='Large')
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel("Cohen's d (Effect Size)", fontsize=12)
        plt.title('Pairwise Effect Sizes Across Rounds', fontsize=14, fontweight='bold')
        plt.legend(fontsize=9, loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'pairwise_effect_sizes_across_rounds.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Statistical test plots saved to {self.save_dir}")
    
    def plot_accuracy_loss_similarity(self):
        """رسم نمودارهای دقت، loss و similarity"""
        print("\nPlotting accuracy, loss and similarity results...")
        
        rounds = list(range(1, self.config.num_rounds + 1))
        
        # 1. نمودار دقت سه الگوریتم در کنار هم
        plt.figure(figsize=(16, 10))
        
        plt.plot(rounds, self.dfl_round_mean, 'b-', linewidth=3, marker='o', markersize=8, label='DFL-DGA')
        plt.fill_between(rounds, 
                        self.dfl_round_mean - self.dfl_round_std,
                        self.dfl_round_mean + self.dfl_round_std,
                        alpha=0.2, color='blue')
        
        plt.plot(rounds, self.fedavg_round_mean, 'r-', linewidth=3, marker='s', markersize=8, label='FedAvg P2P')
        plt.fill_between(rounds,
                        self.fedavg_round_mean - self.fedavg_round_std,
                        self.fedavg_round_mean + self.fedavg_round_std,
                        alpha=0.2, color='red')
        
        plt.plot(rounds, self.wafl_round_mean, 'g-', linewidth=3, marker='^', markersize=8, label='WAFL')
        plt.fill_between(rounds,
                        self.wafl_round_mean - self.wafl_round_std,
                        self.wafl_round_mean + self.wafl_round_std,
                        alpha=0.2, color='green')
        
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title('Accuracy Comparison: DFL-DGA vs FedAvg P2P vs WAFL', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'accuracy_comparison_all.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. نمودار loss سه الگوریتم در کنار هم
        plt.figure(figsize=(16, 10))
        
        plt.plot(rounds, self.dfl_loss_round_mean, 'b-', linewidth=3, marker='o', markersize=8, label='DFL-DGA')
        plt.fill_between(rounds, 
                        self.dfl_loss_round_mean - self.dfl_loss_round_std,
                        self.dfl_loss_round_mean + self.dfl_loss_round_std,
                        alpha=0.2, color='blue')
        
        plt.plot(rounds, self.fedavg_loss_round_mean, 'r-', linewidth=3, marker='s', markersize=8, label='FedAvg P2P')
        plt.fill_between(rounds,
                        self.fedavg_loss_round_mean - self.fedavg_loss_round_std,
                        self.fedavg_loss_round_mean + self.fedavg_loss_round_std,
                        alpha=0.2, color='red')
        
        plt.plot(rounds, self.wafl_loss_round_mean, 'g-', linewidth=3, marker='^', markersize=8, label='WAFL')
        plt.fill_between(rounds,
                        self.wafl_loss_round_mean - self.wafl_loss_round_std,
                        self.wafl_loss_round_mean + self.wafl_loss_round_std,
                        alpha=0.2, color='green')
        
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Loss Comparison: DFL-DGA vs FedAvg P2P vs WAFL', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_comparison_all.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. نمودار similarity سه الگوریتم در کنار هم
        if hasattr(self, 'dfl_sim_mean') and hasattr(self, 'fedavg_sim_mean') and hasattr(self, 'wafl_sim_mean'):
            plt.figure(figsize=(16, 10))
            
            plt.plot(rounds, self.dfl_sim_mean, 'b-', linewidth=3, marker='o', markersize=8, label='DFL-DGA')
            plt.fill_between(rounds, 
                            self.dfl_sim_mean - self.dfl_sim_std,
                            self.dfl_sim_mean + self.dfl_sim_std,
                            alpha=0.2, color='blue')
            
            plt.plot(rounds, self.fedavg_sim_mean, 'r-', linewidth=3, marker='s', markersize=8, label='FedAvg P2P')
            plt.fill_between(rounds,
                            self.fedavg_sim_mean - self.fedavg_sim_std,
                            self.fedavg_sim_mean + self.fedavg_sim_std,
                            alpha=0.2, color='red')
            
            plt.plot(rounds, self.wafl_sim_mean, 'g-', linewidth=3, marker='^', markersize=8, label='WAFL')
            plt.fill_between(rounds,
                            self.wafl_sim_mean - self.wafl_sim_std,
                            self.wafl_sim_mean + self.wafl_sim_std,
                            alpha=0.2, color='green')
            
            plt.xlabel('Round', fontsize=14)
            plt.ylabel('Model Similarity', fontsize=14)
            plt.title('Model Similarity Comparison: DFL-DGA vs FedAvg P2P vs WAFL', fontsize=16, fontweight='bold')
            plt.legend(fontsize=12, loc='lower right')
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1])
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'similarity_comparison_all.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. نمودار ترکیبی دقت، loss و similarity
        fig, axes = plt.subplots(3, 1, figsize=(16, 18))
        
        # دقت
        axes[0].plot(rounds, self.dfl_round_mean, 'b-', linewidth=2, marker='o', markersize=6, label='DFL-DGA')
        axes[0].plot(rounds, self.fedavg_round_mean, 'r-', linewidth=2, marker='s', markersize=6, label='FedAvg')
        axes[0].plot(rounds, self.wafl_round_mean, 'g-', linewidth=2, marker='^', markersize=6, label='WAFL')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
        
        # Loss
        axes[1].plot(rounds, self.dfl_loss_round_mean, 'b-', linewidth=2, marker='o', markersize=6, label='DFL-DGA')
        axes[1].plot(rounds, self.fedavg_loss_round_mean, 'r-', linewidth=2, marker='s', markersize=6, label='FedAvg')
        axes[1].plot(rounds, self.wafl_loss_round_mean, 'g-', linewidth=2, marker='^', markersize=6, label='WAFL')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Comparison', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Similarity
        if hasattr(self, 'dfl_sim_mean'):
            axes[2].plot(rounds, self.dfl_sim_mean, 'b-', linewidth=2, marker='o', markersize=6, label='DFL-DGA')
            axes[2].plot(rounds, self.fedavg_sim_mean, 'r-', linewidth=2, marker='s', markersize=6, label='FedAvg')
            axes[2].plot(rounds, self.wafl_sim_mean, 'g-', linewidth=2, marker='^', markersize=6, label='WAFL')
            axes[2].set_xlabel('Round')
            axes[2].set_ylabel('Similarity')
            axes[2].set_title('Model Similarity Comparison', fontsize=12, fontweight='bold')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'accuracy_loss_similarity_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. نمودار تعداد اعضای مجموعه برتری‌ها (فقط برای DFL-DGA)
        if hasattr(self, 'dominating_round_mean') and hasattr(self, 'dominating_round_std'):
            plt.figure(figsize=(14, 8))
            
            plt.plot(rounds, self.dominating_round_mean, 'purple', linewidth=3, marker='D', markersize=8, 
                    label='Mean Dominating Set Size')
            plt.fill_between(rounds,
                           self.dominating_round_mean - self.dominating_round_std,
                           self.dominating_round_mean + self.dominating_round_std,
                           alpha=0.2, color='purple', label='±1 Std Dev')
            
            plt.xlabel('Round', fontsize=14)
            plt.ylabel('Dominating Set Size', fontsize=14)
            plt.title('Average Dominating Set Size in DFL-DGA', fontsize=16, fontweight='bold')
            plt.legend(fontsize=12, loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'dominating_set_size.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 6. نمودار ترکیبی دقت و اندازه مجموعه برتری
            fig, ax1 = plt.subplots(figsize=(18, 12))
            
            # محور y اول: دقت
            ax1.plot(rounds, self.dfl_round_mean, 'b-', linewidth=3, marker='o', markersize=8, label='DFL-DGA Accuracy')
            ax1.fill_between(rounds, 
                            self.dfl_round_mean - self.dfl_round_std,
                            self.dfl_round_mean + self.dfl_round_std,
                            alpha=0.2, color='blue')
            
            ax1.plot(rounds, self.fedavg_round_mean, 'r-', linewidth=3, marker='s', markersize=8, label='FedAvg Accuracy')
            ax1.fill_between(rounds,
                            self.fedavg_round_mean - self.fedavg_round_std,
                            self.fedavg_round_mean + self.fedavg_round_std,
                            alpha=0.2, color='red')
            
            ax1.plot(rounds, self.wafl_round_mean, 'g-', linewidth=3, marker='^', markersize=8, label='WAFL Accuracy')
            ax1.fill_between(rounds,
                            self.wafl_round_mean - self.wafl_round_std,
                            self.wafl_round_mean + self.wafl_round_std,
                            alpha=0.2, color='green')
            
            ax1.set_xlabel('Round', fontsize=14)
            ax1.set_ylabel('Accuracy', fontsize=14, color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.set_ylim([0, 1])
            ax1.grid(True, alpha=0.3)
            
            # محور y دوم: اندازه مجموعه برتری
            ax2 = ax1.twinx()
            ax2.plot(rounds, self.dominating_round_mean, 'purple', linewidth=2, linestyle='--', marker='D', markersize=6, 
                    label='Dominating Set Size')
            ax2.fill_between(rounds,
                            self.dominating_round_mean - self.dominating_round_std,
                            self.dominating_round_mean + self.dominating_round_std,
                            alpha=0.1, color='purple')
            
            ax2.set_ylabel('Dominating Set Size', fontsize=14, color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
            
            # اضافه کردن legend برای هر دو محور
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
            
            plt.title('Accuracy and Dominating Set Size Comparison', fontsize=18, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'accuracy_and_dominating_set_all.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 7. نمودار دقت هر گره در آخرین راند
        if hasattr(self, 'dfl_node_round_mean') and hasattr(self, 'fedavg_node_round_mean') and hasattr(self, 'wafl_node_round_mean'):
            final_round_idx = self.config.num_rounds - 1
            
            nodes = list(range(1, self.config.num_nodes + 1))
            dfl_final_acc = self.dfl_node_round_mean[final_round_idx]
            fedavg_final_acc = self.fedavg_node_round_mean[final_round_idx]
            wafl_final_acc = self.wafl_node_round_mean[final_round_idx]
            
            x = np.arange(len(nodes))
            width = 0.25
            
            plt.figure(figsize=(16, 10))
            
            plt.bar(x - width, dfl_final_acc, width, label='DFL-DGA', alpha=0.8, color='blue')
            plt.bar(x, fedavg_final_acc, width, label='FedAvg', alpha=0.8, color='red')
            plt.bar(x + width, wafl_final_acc, width, label='WAFL', alpha=0.8, color='green')
            
            plt.xlabel('Node ID', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.title(f'Final Round Accuracy per Node (Round {self.config.num_rounds})', fontsize=16, fontweight='bold')
            plt.xticks(x, nodes)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'final_round_node_accuracies_all.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 8. نمودار مقایسه‌ای بین الگوریتم‌ها در دورهای مختلف
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # دور اول
        axes[0, 0].bar(['DFL-DGA', 'FedAvg', 'WAFL'], 
                      [self.dfl_round_mean[0], self.fedavg_round_mean[0], self.wafl_round_mean[0]],
                      color=['blue', 'red', 'green'], alpha=0.7)
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Round 1', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # دور میانی
        mid_round = self.config.num_rounds // 2
        axes[0, 1].bar(['DFL-DGA', 'FedAvg', 'WAFL'], 
                      [self.dfl_round_mean[mid_round], self.fedavg_round_mean[mid_round], self.wafl_round_mean[mid_round]],
                      color=['blue', 'red', 'green'], alpha=0.7)
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title(f'Round {mid_round+1}', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # دور پایانی
        final_round = self.config.num_rounds - 1
        axes[1, 0].bar(['DFL-DGA', 'FedAvg', 'WAFL'], 
                      [self.dfl_round_mean[final_round], self.fedavg_round_mean[final_round], self.wafl_round_mean[final_round]],
                      color=['blue', 'red', 'green'], alpha=0.7)
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title(f'Final Round ({self.config.num_rounds})', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # نمودار پیشرفت از دور اول به آخر
        progress_data = [
            (self.dfl_round_mean[final_round] - self.dfl_round_mean[0]) / self.dfl_round_mean[0] * 100 if self.dfl_round_mean[0] > 0 else 0,
            (self.fedavg_round_mean[final_round] - self.fedavg_round_mean[0]) / self.fedavg_round_mean[0] * 100 if self.fedavg_round_mean[0] > 0 else 0,
            (self.wafl_round_mean[final_round] - self.wafl_round_mean[0]) / self.wafl_round_mean[0] * 100 if self.wafl_round_mean[0] > 0 else 0
        ]
        
        axes[1, 1].bar(['DFL-DGA', 'FedAvg', 'WAFL'], progress_data,
                      color=['blue', 'red', 'green'], alpha=0.7)
        axes[1, 1].set_ylabel('Improvement (%)')
        axes[1, 1].set_title('Percentage Improvement from First to Last Round', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'algorithm_comparison_grid.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Accuracy, loss and similarity plots saved to {self.save_dir}")
    
    def save_final_results(self):
        """ذخیره نتایج نهایی"""
        final_results = {
            'config': {
                'dataset': str(self.config.dataset),
                'num_nodes': int(self.config.num_nodes),
                'num_rounds': int(self.config.num_rounds),
                'num_runs': int(self.config.num_runs),
                'mobility_model': str(self.config.mobility_model),
                'distance_method': str(self.config.distance_method),
                'dirichlet_alpha': float(self.config.dirichlet_alpha),
                'wafl_lambda': float(self.config.wafl_lambda),
                'dfl_beta': float(self.config.dfl_beta)
            },
            'descriptive_statistics': {
                'dfl_dga': {
                    'mean_accuracy_per_round': [float(x) for x in self.dfl_round_mean.tolist()],
                    'std_accuracy_per_round': [float(x) for x in self.dfl_round_std.tolist()],
                    'mean_loss_per_round': [float(x) for x in self.dfl_loss_round_mean.tolist()],
                    'std_loss_per_round': [float(x) for x in self.dfl_loss_round_std.tolist()],
                    'node_round_mean': [[float(y) for y in x] for x in self.dfl_node_round_mean.tolist()],
                    'node_round_std': [[float(y) for y in x] for x in self.dfl_node_round_std.tolist()],
                    'final_accuracy_mean': float(self.dfl_round_mean[-1]) if len(self.dfl_round_mean) > 0 else 0,
                    'final_accuracy_std': float(self.dfl_round_std[-1]) if len(self.dfl_round_std) > 0 else 0,
                    'final_loss_mean': float(self.dfl_loss_round_mean[-1]) if len(self.dfl_loss_round_mean) > 0 else 0,
                    'final_loss_std': float(self.dfl_loss_round_std[-1]) if len(self.dfl_loss_round_std) > 0 else 0
                },
                'fedavg_p2p': {
                    'mean_accuracy_per_round': [float(x) for x in self.fedavg_round_mean.tolist()],
                    'std_accuracy_per_round': [float(x) for x in self.fedavg_round_std.tolist()],
                    'mean_loss_per_round': [float(x) for x in self.fedavg_loss_round_mean.tolist()],
                    'std_loss_per_round': [float(x) for x in self.fedavg_loss_round_std.tolist()],
                    'node_round_mean': [[float(y) for y in x] for x in self.fedavg_node_round_mean.tolist()],
                    'node_round_std': [[float(y) for y in x] for x in self.fedavg_node_round_std.tolist()],
                    'final_accuracy_mean': float(self.fedavg_round_mean[-1]) if len(self.fedavg_round_mean) > 0 else 0,
                    'final_accuracy_std': float(self.fedavg_round_std[-1]) if len(self.fedavg_round_std) > 0 else 0,
                    'final_loss_mean': float(self.fedavg_loss_round_mean[-1]) if len(self.fedavg_loss_round_mean) > 0 else 0,
                    'final_loss_std': float(self.fedavg_loss_round_std[-1]) if len(self.fedavg_loss_round_std) > 0 else 0
                },
                'wafl': {
                    'mean_accuracy_per_round': [float(x) for x in self.wafl_round_mean.tolist()],
                    'std_accuracy_per_round': [float(x) for x in self.wafl_round_std.tolist()],
                    'mean_loss_per_round': [float(x) for x in self.wafl_loss_round_mean.tolist()],
                    'std_loss_per_round': [float(x) for x in self.wafl_loss_round_std.tolist()],
                    'node_round_mean': [[float(y) for y in x] for x in self.wafl_node_round_mean.tolist()],
                    'node_round_std': [[float(y) for y in x] for x in self.wafl_node_round_std.tolist()],
                    'final_accuracy_mean': float(self.wafl_round_mean[-1]) if len(self.wafl_round_mean) > 0 else 0,
                    'final_accuracy_std': float(self.wafl_round_std[-1]) if len(self.wafl_round_std) > 0 else 0,
                    'final_loss_mean': float(self.wafl_loss_round_mean[-1]) if len(self.wafl_loss_round_mean) > 0 else 0,
                    'final_loss_std': float(self.wafl_loss_round_std[-1]) if len(self.wafl_loss_round_std) > 0 else 0
                }
            },
            'statistical_tests': self.statistical_results,
            'summary': {
                'final_round_comparison': self.statistical_results.get('final_round_tests', {}),
                'interpretation': str(self.generate_summary_interpretation())
            }
        }
        
        # اضافه کردن آمار similarity
        if hasattr(self, 'dfl_sim_mean'):
            final_results['similarity_statistics'] = {
                'dfl_dga': {
                    'mean_similarity_per_round': [float(x) for x in self.dfl_sim_mean.tolist()],
                    'std_similarity_per_round': [float(x) for x in self.dfl_sim_std.tolist()],
                    'final_similarity_mean': float(self.dfl_sim_mean[-1]) if len(self.dfl_sim_mean) > 0 else 0
                },
                'fedavg_p2p': {
                    'mean_similarity_per_round': [float(x) for x in self.fedavg_sim_mean.tolist()],
                    'std_similarity_per_round': [float(x) for x in self.fedavg_sim_std.tolist()],
                    'final_similarity_mean': float(self.fedavg_sim_mean[-1]) if len(self.fedavg_sim_mean) > 0 else 0
                },
                'wafl': {
                    'mean_similarity_per_round': [float(x) for x in self.wafl_sim_mean.tolist()],
                    'std_similarity_per_round': [float(x) for x in self.wafl_sim_std.tolist()],
                    'final_similarity_mean': float(self.wafl_sim_mean[-1]) if len(self.wafl_sim_mean) > 0 else 0
                }
            }
        
        # اضافه کردن آمار مجموعه برتری اگر موجود باشد
        if hasattr(self, 'dominating_round_mean'):
            final_results['dominating_set_statistics'] = {
                'mean_size_per_round': [float(x) for x in self.dominating_round_mean.tolist()],
                'std_size_per_round': [float(x) for x in self.dominating_round_std.tolist()],
                'final_mean_size': float(self.dominating_round_mean[-1]) if len(self.dominating_round_mean) > 0 else 0
            }
        
        with open(os.path.join(self.save_dir, 'final_statistical_results.json'), 'w') as f:
            json.dump(final_results, f, cls=JSONEncoder, indent=4)
        
        # ذخیره خلاصه نتایج به صورت CSV
        self.save_summary_csv()
        
        print(f"\nFinal results saved to {self.save_dir}")
    
    def save_summary_csv(self):
        """ذخیره خلاصه نتایج به صورت CSV"""
        summary_data = []
        
        for round_idx, test in enumerate(self.statistical_results['per_round_tests']):
            row_data = {
                'round': int(round_idx + 1),
                'dfl_accuracy_mean': float(test['descriptive_stats']['dfl_mean']),
                'dfl_accuracy_std': float(test['descriptive_stats']['dfl_std']),
                'fedavg_accuracy_mean': float(test['descriptive_stats']['fedavg_mean']),
                'fedavg_accuracy_std': float(test['descriptive_stats']['fedavg_std']),
                'wafl_accuracy_mean': float(test['descriptive_stats']['wafl_mean']),
                'wafl_accuracy_std': float(test['descriptive_stats']['wafl_std'])
            }
            
            # اضافه کردن loss
            if hasattr(self, 'dfl_loss_round_mean'):
                row_data['dfl_loss_mean'] = float(self.dfl_loss_round_mean[round_idx])
                row_data['dfl_loss_std'] = float(self.dfl_loss_round_std[round_idx])
                row_data['fedavg_loss_mean'] = float(self.fedavg_loss_round_mean[round_idx])
                row_data['fedavg_loss_std'] = float(self.fedavg_loss_round_std[round_idx])
                row_data['wafl_loss_mean'] = float(self.wafl_loss_round_mean[round_idx])
                row_data['wafl_loss_std'] = float(self.wafl_loss_round_std[round_idx])
            
            # اضافه کردن similarity
            if hasattr(self, 'dfl_sim_mean'):
                row_data['dfl_similarity_mean'] = float(self.dfl_sim_mean[round_idx])
                row_data['dfl_similarity_std'] = float(self.dfl_sim_std[round_idx])
                row_data['fedavg_similarity_mean'] = float(self.fedavg_sim_mean[round_idx])
                row_data['fedavg_similarity_std'] = float(self.fedavg_sim_std[round_idx])
                row_data['wafl_similarity_mean'] = float(self.wafl_sim_mean[round_idx])
                row_data['wafl_similarity_std'] = float(self.wafl_sim_std[round_idx])
            
            # اضافه کردن مقایسه‌های زوجی
            for comparison in test.get('pairwise_comparisons', []):
                comp_name = comparison['comparison'].replace(' ', '_').replace('-', '_').lower()
                row_data[f'{comp_name}_p_value'] = float(comparison['p_value'])
                row_data[f'{comp_name}_cohens_d'] = float(comparison['cohens_d'])
                row_data[f'{comp_name}_significant_0_05'] = bool(comparison['p_value'] < 0.05)
            
            # اضافه کردن آمار مجموعه برتری اگر موجود باشد
            if hasattr(self, 'dominating_round_mean'):
                row_data['dominating_set_mean'] = float(self.dominating_round_mean[round_idx])
                row_data['dominating_set_std'] = float(self.dominating_round_std[round_idx])
            
            summary_data.append(row_data)
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(os.path.join(self.save_dir, 'statistical_summary.csv'), index=False)
        
        # ذخیره رتبه‌بندی نهایی
        final_ranking = {
            'algorithm': ['DFL-DGA', 'FedAvg', 'WAFL'],
            'final_accuracy': [
                float(self.dfl_round_mean[-1]) if len(self.dfl_round_mean) > 0 else 0,
                float(self.fedavg_round_mean[-1]) if len(self.fedavg_round_mean) > 0 else 0,
                float(self.wafl_round_mean[-1]) if len(self.wafl_round_mean) > 0 else 0
            ],
            'final_accuracy_std': [
                float(self.dfl_round_std[-1]) if len(self.dfl_round_std) > 0 else 0,
                float(self.fedavg_round_std[-1]) if len(self.fedavg_round_std) > 0 else 0,
                float(self.wafl_round_std[-1]) if len(self.wafl_round_std) > 0 else 0
            ],
            'final_loss': [
                float(self.dfl_loss_round_mean[-1]) if len(self.dfl_loss_round_mean) > 0 else 0,
                float(self.fedavg_loss_round_mean[-1]) if len(self.fedavg_loss_round_mean) > 0 else 0,
                float(self.wafl_loss_round_mean[-1]) if len(self.wafl_loss_round_mean) > 0 else 0
            ],
            'final_loss_std': [
                float(self.dfl_loss_round_std[-1]) if len(self.dfl_loss_round_std) > 0 else 0,
                float(self.fedavg_loss_round_std[-1]) if len(self.fedavg_loss_round_std) > 0 else 0,
                float(self.wafl_loss_round_std[-1]) if len(self.wafl_loss_round_std) > 0 else 0
            ]
        }
        
        if hasattr(self, 'dfl_sim_mean'):
            final_ranking['final_similarity'] = [
                float(self.dfl_sim_mean[-1]) if len(self.dfl_sim_mean) > 0 else 0,
                float(self.fedavg_sim_mean[-1]) if len(self.fedavg_sim_mean) > 0 else 0,
                float(self.wafl_sim_mean[-1]) if len(self.wafl_sim_mean) > 0 else 0
            ]
            final_ranking['final_similarity_std'] = [
                float(self.dfl_sim_std[-1]) if len(self.dfl_sim_std) > 0 else 0,
                float(self.fedavg_sim_std[-1]) if len(self.fedavg_sim_std) > 0 else 0,
                float(self.wafl_sim_std[-1]) if len(self.wafl_sim_std) > 0 else 0
            ]
        
        df_ranking = pd.DataFrame(final_ranking)
        df_ranking = df_ranking.sort_values('final_accuracy', ascending=False)
        df_ranking['rank'] = range(1, len(df_ranking) + 1)
        df_ranking.to_csv(os.path.join(self.save_dir, 'final_ranking.csv'), index=False)
    
    def generate_summary_interpretation(self):
        """تولید تفسیر خلاصه از نتایج آماری"""
        final_test = self.statistical_results.get('final_round_tests', {})
        
        if not final_test:
            return "No final round test results available."
        
        interpretation = []
        
        interpretation.append("نتایج نهایی:")
        interpretation.append(f"میانگین دقت در دور آخر:")
        interpretation.append(f"  DFL-DGA: {final_test['descriptive_stats']['dfl_mean']:.4f} (±{final_test['descriptive_stats']['dfl_std']:.4f})")
        interpretation.append(f"  FedAvg: {final_test['descriptive_stats']['fedavg_mean']:.4f} (±{final_test['descriptive_stats']['fedavg_std']:.4f})")
        interpretation.append(f"  WAFL: {final_test['descriptive_stats']['wafl_mean']:.4f} (±{final_test['descriptive_stats']['wafl_std']:.4f})")
        
        interpretation.append(f"\nمیانگین loss در دور آخر:")
        if hasattr(self, 'dfl_loss_round_mean'):
            interpretation.append(f"  DFL-DGA: {self.dfl_loss_round_mean[-1]:.4f} (±{self.dfl_loss_round_std[-1]:.4f})")
            interpretation.append(f"  FedAvg: {self.fedavg_loss_round_mean[-1]:.4f} (±{self.fedavg_loss_round_std[-1]:.4f})")
            interpretation.append(f"  WAFL: {self.wafl_loss_round_mean[-1]:.4f} (±{self.wafl_loss_round_std[-1]:.4f})")
        
        interpretation.append("\nمقایسه‌های آماری (دور آخر):")
        
        for comparison in final_test.get('pairwise_comparisons', []):
            p_value = comparison['p_value']
            cohens_d = comparison['cohens_d']
            effect_interpretation = self.interpret_cohens_d(cohens_d)
            
            if p_value < 0.01:
                sig_text = "معنی‌دار در سطح 0.01"
            elif p_value < 0.05:
                sig_text = "معنی‌دار در سطح 0.05"
            else:
                sig_text = "غیرمعنی‌دار"
            
            interpretation.append(f"  {comparison['comparison']}: p={p_value:.4f} ({sig_text}), Cohen's d={cohens_d:.3f} ({effect_interpretation})")
        
        # تعیین بهترین الگوریتم
        algorithms = [
            ('DFL-DGA', final_test['descriptive_stats']['dfl_mean']),
            ('FedAvg', final_test['descriptive_stats']['fedavg_mean']),
            ('WAFL', final_test['descriptive_stats']['wafl_mean'])
        ]
        
        best_algorithm = max(algorithms, key=lambda x: x[1])
        interpretation.append(f"\nبهترین الگوریتم: {best_algorithm[0]} با دقت {best_algorithm[1]:.4f}")
        
        # اطلاعات similarity
        if hasattr(self, 'dfl_sim_mean') and len(self.dfl_sim_mean) > 0:
            interpretation.append(f"\nمیانگین similarity در دور آخر:")
            interpretation.append(f"  DFL-DGA: {self.dfl_sim_mean[-1]:.4f}")
            interpretation.append(f"  FedAvg: {self.fedavg_sim_mean[-1]:.4f}")
            interpretation.append(f"  WAFL: {self.wafl_sim_mean[-1]:.4f}")
        
        # اطلاعات مجموعه برتری
        if hasattr(self, 'dominating_round_mean') and len(self.dominating_round_mean) > 0:
            avg_dominating_size = self.dominating_round_mean[-1]
            interpretation.append(f"\nمیانگین اندازه مجموعه برتری در DFL-DGA: {avg_dominating_size:.2f} گره")
        
        return "\n".join(interpretation)
