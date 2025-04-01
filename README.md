# Telegram Passcode Cracker - Military Grade Edition

![Version](https://img.shields.io/badge/version-3.3.0-red.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

## ⚠️ LEGAL DISCLAIMER

**THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL AND SECURITY TESTING PURPOSES ONLY. MISUSE OF THIS SOFTWARE MAY CONSTITUTE A CRIMINAL OFFENSE IN MANY JURISDICTIONS.**

Using this software to access accounts without explicit authorization from the owner is illegal and may result in severe legal consequences.

## 📋 Overview

A high-performance security testing tool designed to evaluate the resilience of Telegram passcodes against brute-force attacks. Employs advanced proxy management and multi-threading to maximize attempt efficiency.

## ✨ Key Features

- **Maximum Performance**: Optimized multi-threaded architecture with dynamic resource allocation
- **Advanced Proxy Management**: 
  - Multi-protocol support (SOCKS5, SOCKS4, HTTP)
  - Automatic validation and performance ranking
  - Intelligent rotation and failure recovery
- **Real-time Interface**: Comprehensive statistics and progress monitoring
- **Adaptive Optimization**: Self-tunes based on system capabilities and network conditions
- **Intelligent Code Prioritization**: Tests common patterns first (sequential, repeated digits)
- **Comprehensive Logging System**: Detailed logs for in-depth analysis

## 🔧 Requirements

- Python 3.8+
- Required packages: requests, termcolor, rich
- Internet connection
- Proxy list (recommended)

## 📦 Installation

```bash
git clone https://github.com/AI4Arabs/telegram-passcode-cracker.git
cd telegram-passcode-cracker
pip install -r requirements.txt
```

## 🚀 Usage

1. Prepare `proxy.txt` with proxies (format: `IP:PORT` or `IP:PORT:LOCATION`)
2. Prepare `numbers.txt` with passcodes to try
3. Run: `python main.py`
4. Enter target phone number when prompted (+COUNTRYCODE format)

## ⚙️ Configuration

Create `config.json` to customize:

```json
{
    "MAX_THREADS": 500,
    "REQUEST_TIMEOUT": 6,
    "PROXY_PROTOCOLS": ["socks5", "socks4", "http"],
    "PROXY_FALLBACK_MODE": true,
    "PROXY_CHECK_TIMEOUT": 5
}
```

## 📄 File Structure

- `main.py`: Main executable
- `proxy.txt`: Proxy list
- `numbers.txt`: Passcodes to test
- `config.json`: Custom configuration
- `logs/`: Detailed logs directory
- `cracked.txt`: Successful results

## 🔍 Troubleshooting

- **No valid proxies**: Check proxy.txt format is correct
- **Slow performance**: Increase MAX_THREADS or use faster proxies
- **Connection errors**: Check internet connectivity and proxy status

## 📜 Copyright

© 2025 [AI4Arabs](https://t.me/AI4Arabs). All rights reserved.

**Developed by [AI4Arabs](https://t.me/AI4Arabs) team**

---

**NOTE**: This tool is intended exclusively for security professionals, researchers, and account owners testing their own accounts' security. The authors assume no responsibility for misuse.
