<<<<<<< HEAD
"""
ChronaNet Framework
Copyright (c) 2025 Luis Morató de Dalmases

This software is licensed under the MIT License for non-commercial use.
Commercial use requires a separate license: contact morato.lluis@gmail.com.
See LICENSE.md for full details: https://github.com/ChronaNet

Attribution:
ChronaCoin Protocol, developed by Luis Morató de Dalmases (2025).
Powered by quantum-temporal cryptography and chrono-algorithmic architecture.
"""

import numpy as np
import json
from scipy.fft import fft
import logging
from datetime import datetime
import hashlib
from flask import Flask, request, jsonify
import requests
import threading
import streamlit as st
import time

# Configuració de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TAU = 0.37037  # Període temporal base (segons)
COHERENCE_THRESHOLD = 0.707  # Llindar de coherència per PoC
PHASE_TOLERANCE = 1e-2  # Tolerància de fase (radians)
REWARD_BASE = 10.0  # Recompensa base en Chronacoins per bloc validat

class Tesseract:
    def __init__(self, id, triad, phase, intensity):
        self.id = id
        self.triad = triad
        self.phase = phase
        self.intensity = intensity
        self.omega = self._compute_omega()

    def _compute_omega(self):
        rad_val = self._radical(*self.triad)
        return 2 * np.pi / (TAU / np.log(rad_val) if rad_val > 1 else TAU)

    def _radical(self, a, b, c):
        def prime_factors(n):
            factors = set()
            d = 2
            while n > 1:
                if n % d == 0:
                    factors.add(d)
                    n //= d
                d += 1
                if d * d > n and n > 1:
                    factors.add(n)
                    break
            return factors
        return np.prod(list(prime_factors(a) | prime_factors(b) | prime_factors(c)))

    def psi(self, t):
        return self.intensity * np.exp(1j * (self.omega * t + self.phase))

class CronaWallet:
    def __init__(self, tesseract_t8, wallet_file=None):
        self.t8 = tesseract_t8
        self.phase_key = self._compute_phase_key()
        self.wallet_file = wallet_file or f"wallet_{tesseract_t8.id}.json"
        self.balance = 0.0
        self.history = []
        self.load_wallet()

    def _compute_phase_key(self, t_max=1.0, dt=0.01):
        t_vec = np.arange(0, t_max, dt)
        psi_vec = [self.t8.psi(t) for t in t_vec]
        return np.trapz(psi_vec, t_vec)

    def sign_transaction(self, txn, t):
        txn['phase_key'] = abs(self.phase_key)
        return txn

    def update_balance(self, amount, txn_type, txn):
        self.balance += amount
        self.history.append({
            "type": txn_type,
            "amount": amount,
            "txn": txn,
            "timestamp": datetime.now().timestamp()
        })
        self.save_wallet()

    def save_wallet(self):
        try:
            with open(self.wallet_file, 'w') as f:
                json.dump({"balance": self.balance, "history": self.history}, f, indent=2)
            logger.info(f"Wallet guardat a {self.wallet_file}")
        except IOError as e:
            logger.error(f"Error guardant wallet: {e}")

    def load_wallet(self):
        try:
            with open(self.wallet_file, 'r') as f:
                data = json.load(f)
                self.balance = data.get("balance", 0.0)
                self.history = data.get("history", [])
            logger.info(f"Wallet carregat des de {self.wallet_file}")
        except FileNotFoundError:
            logger.info(f"No s'ha trobat {self.wallet_file}. Iniciant wallet buit.")

class ChronaBlockChain:
    def __init__(self, node_id):
        self.chain = []
        self.current_transactions = []
        self.node_id = node_id
        self.ledger_path = f"ledger_examples/blockchain_{node_id}.json"
        self.load_blockchain()

    def add_transaction(self, txn):
        self.current_transactions.append(txn)

    def create_block(self, t, coherence, t8_hash, prev_hash="0"):
        block = {
            "block_id": len(self.chain) + 1,
            "timestamp": t,
            "transactions": self.current_transactions,
            "coherence": coherence,
            "t8_hash": list(np.abs(t8_hash[:10])),
            "prev_hash": prev_hash
        }
        block["block_hash"] = hashlib.sha256(json.dumps(block, sort_keys=True).encode()).hexdigest()
        self.chain.append(block)
        self.current_transactions = []
        self.save_blockchain()
        return block

    def save_blockchain(self):
        try:
            with open(self.ledger_path, 'w') as f:
                json.dump(self.chain, f, indent=2)
            logger.info(f"Blockchain guardat a {self.ledger_path}")
        except IOError as e:
            logger.error(f"Error guardant blockchain: {e}")

    def load_blockchain(self):
        try:
            with open(self.ledger_path, 'r') as f:
                self.chain = json.load(f)
            logger.info(f"Blockchain carregat des de {self.ledger_path}")
        except FileNotFoundError:
            logger.info(f"No s'ha trobat {self.ledger_path}. Iniciant blockchain buit.")

    def validate_block(self, block):
        expected_hash = hashlib.sha256(json.dumps({k: block[k] for k in block if k != 'block_hash'}, sort_keys=True).encode()).hexdigest()
        return block["block_hash"] == expected_hash and block["coherence"] >= COHERENCE_THRESHOLD

class ChronaNetNode:
    def __init__(self, node_id, tesseracts, port, tavari_db='tavari_symbols.json'):
        self.node_id = node_id
        self.tesseracts = tesseracts
        self.t8 = next(tess for tess in tesseracts if tess.id == 8)
        self.wallet = CronaWallet(self.t8, f"ledger_examples/wallet_{node_id}.json")
        self.tavari_db = self._load_tavari_db(tavari_db)
        self.phase_vector = None
        self.is_active = True
        self.blockchain = ChronaBlockChain(node_id)
        self.ledger_path = f"ledger_examples/ledger_{node_id}.json"
        self.port = port
        self.peers = []
        logger.info(f"Node {node_id} inicialitzat amb {len(tesseracts)} tesseracts al port {port}.")
        self.sync_chain()

    def _load_tavari_db(self, tavari_file):
        default_db = {
            "Hg-Ax-Ro": {"value": 123, "freq_mod": 7.83},
            "Li-St-Tx": {"value": 456, "freq_mod": 11.7}
        }
        try:
            with open(tavari_file, 'w') as f:
                json.dump(default_db, f)
            with open(tavari_file, 'r') as f:
                return json.load(f)
        except IOError:
            logger.warning("Error d'E/S amb tavari_symbols.json. Usant base de dades per defecte.")
            return default_db

    def save_transaction(self, txn, valid=True):
        record = txn.copy()
        record["valid"] = valid
        try:
            with open(self.ledger_path, 'r') as f:
                ledger = json.load(f)
        except FileNotFoundError:
            ledger = []
        ledger.append(record)
        with open(self.ledger_path, 'w') as f:
            json.dump(ledger, f, indent=2)

    def resonance_field(self, t):
        R = 0
        for i, tess_i in enumerate(self.tesseracts):
            for j, tess_j in enumerate(self.tesseracts[i+1:], i+1):
                chi_ij = np.log(tess_i._radical(*tess_i.triad)) / np.linalg.norm(
                    np.array(tess_i.triad) - np.array(tess_j.triad)
                )
                delta_phi = tess_i.phase - tess_j.phase + (tess_i.omega - tess_j.omega) * t
                R += chi_ij * np.cos(delta_phi)
        return R

    def spectral_hash(self, t, dt=0.01):
        t_vec = np.arange(0, t, dt)
        psi_vec = [self.t8.psi(ti) for ti in t_vec]
        return fft(psi_vec)

    def coherence(self, t, network_nodes):
        psi_sum = sum(node.t8.psi(t) for node in network_nodes if node.is_active)
        norm_sum = sum(abs(node.t8.psi(t)) for node in network_nodes if node.is_active)
        C = abs(psi_sum) / norm_sum if norm_sum != 0 else 0
        return C

    def sync_phase(self, t, network_nodes):
        self.phase_vector = np.array([np.cos(self.t8.phase + self.t8.omega * t),
                                      np.sin(self.t8.phase + self.t8.omega * t)])
        for node in network_nodes:
            if node.node_id != self.node_id and node.is_active:
                other_vector = node.phase_vector
                if other_vector is not None:
                    dot_product = np.dot(self.phase_vector, other_vector)
                    if dot_product < np.cos(PHASE_TOLERANCE):
                        self.is_active = False
                        logger.warning(f"Node {self.node_id} desactivat per desincronització de fase.")
                        return False
        return True

    def map_tavari(self, symbol):
        omega = self.t8.omega
        tavari = self.tavari_db.get(symbol, {"value": sum(ord(c) for c in ''.join(symbol))})
        return tavari["value"] % int(omega * 100)

    def encode_message(self, msg, t, tavari):
        R = self.resonance_field(t)
        tavari_mod = self.map_tavari(tavari)
        return ''.join(chr((ord(c) ^ int(t * R)) + tavari_mod) for c in msg)

    def decode_message(self, encoded, t, tavari):
        R = self.resonance_field(t)
        tavari_mod = self.map_tavari(tavari)
        return ''.join(chr(((ord(c) - tavari_mod) ^ int(t * R)) % 256) for c in encoded)  # Correcció i assegurar valors vàlids

    def create_transaction(self, recipient_id, value, t, tavari):
        if self.wallet.balance < value:
            logger.error(f"Saldo insuficient a {self.node_id}: {self.wallet.balance} < {value}")
            return None
        txn = {
            'sender': self.node_id,
            'recipient': recipient_id,
            'triad': self.t8.triad,
            'phi': self.t8.phase,
            'omega': self.t8.omega,
            'value': value,
            'timestamp': t,
            'tavari': tavari,
            'hash': self.spectral_hash(t).tolist()
        }
        txn = self.wallet.sign_transaction(txn, t)
        self.wallet.update_balance(-value, "sent", txn)
        logger.info(f"Transacció creada: {txn['sender']} -> {txn['recipient']}, valor={value}")
        return txn

    def validate_transaction(self, txn, t, network_nodes):
        if not self.is_active:
            logger.error(f"Node {self.node_id} inactiu. Validació fallida.")
            return False
        C = self.coherence(t, network_nodes)
        if C < COHERENCE_THRESHOLD:
            logger.warning(f"Coherència insuficient: C={C:.3f} < {COHERENCE_THRESHOLD}")
            return False
        expected_hash = self.spectral_hash(t)
        hash_diff = abs(np.array(txn['hash']) - expected_hash).sum()
        if hash_diff > 1e-3:
            logger.error(f"Hash espectral no coincident: diff={hash_diff}")
            return False
        phase_key_valid = abs(txn['phase_key'] - abs(self.wallet.phase_key)) < 1e-2
        if not phase_key_valid:
            logger.error("Clau de fase invàlida.")
            return False
        logger.info(f"Transacció validada: {txn['sender']} -> {txn['recipient']}")
        self.save_transaction(txn, valid=True)
        self.blockchain.add_transaction(txn)
        if txn['recipient'] == self.node_id:
            self.wallet.update_balance(txn['value'], "received", txn)
        if len(self.blockchain.current_transactions) >= 3:
            prev_hash = self.blockchain.chain[-1]["block_hash"] if self.blockchain.chain else "0"
            block = self.blockchain.create_block(t, C, self.spectral_hash(t), prev_hash)
            reward = REWARD_BASE * C
            self.wallet.update_balance(reward, "reward", {"block_id": block["block_id"]})
            logger.info(f"Bloc creat per node {self.node_id}: {block['block_id']} amb {len(block['transactions'])} transaccions. Recompensa: {reward} Chronacoins.")
            self.propagate_block(block)
        return True

    def add_peer(self, peer_url):
        if peer_url not in self.peers:
            self.peers.append(peer_url)
            logger.info(f"Peer afegit a {self.node_id}: {peer_url}")

    def sync_chain(self):
        longest_chain = self.blockchain.chain
        max_length = len(longest_chain)
        max_coherence = sum(block["coherence"] for block in longest_chain) if longest_chain else 0
        for peer in self.peers:
            try:
                response = requests.get(f"{peer}/chain")
                if response.status_code == 200:
                    peer_chain = response.json()
                    peer_length = len(peer_chain)
                    peer_coherence = sum(block["coherence"] for block in peer_chain) if peer_chain else 0
                    if peer_length > max_length or (peer_length == max_length and peer_coherence > max_coherence):
                        if all(self.blockchain.validate_block(block) for block in peer_chain):
                            longest_chain = peer_chain
                            max_length = peer_length
                            max_coherence = peer_coherence
            except requests.RequestException as e:
                logger.error(f"Error sincronitzant amb {peer}: {e}")
        if longest_chain != self.blockchain.chain:
            self.blockchain.chain = longest_chain
            self.blockchain.save_blockchain()
            logger.info(f"Blockchain sincronitzat a {self.node_id} amb {len(longest_chain)} blocs.")

    def propagate_block(self, block):
        for peer in self.peers:
            try:
                response = requests.post(f"{peer}/block", json=block)
                if response.status_code == 200:
                    logger.info(f"Bloc propagat a {peer}")
                else:
                    logger.warning(f"Error propagant bloc a {peer}: {response.status_code}")
            except requests.RequestException as e:
                logger.error(f"Error de connexió amb {peer}: {e}")

    def receive_block(self, block):
        if self.blockchain.validate_block(block):
            self.blockchain.chain.append(block)
            self.blockchain.save_blockchain()
            logger.info(f"Bloc rebut i afegit a {self.node_id}: {block['block_id']}")
        else:
            logger.warning(f"Bloc rebut invàlid a {self.node_id}")

def start_node_server(node, host='localhost'):
    app = Flask(__name__)

    @app.route('/transaction', methods=['POST'])
    def receive_transaction():
        txn = request.get_json()
        node.validate_transaction(txn, txn['timestamp'], [node])
        return jsonify({"status": "Transaction received"}), 200

    @app.route('/block', methods=['POST'])
    def receive_block():
        block = request.get_json()
        node.receive_block(block)
        return jsonify({"status": "Block received"}), 200

    @app.route('/chain', methods=['GET'])
    def get_chain():
        return jsonify(node.blockchain.chain), 200

    threading.Thread(target=app.run, kwargs={'host': host, 'port': node.port}).start()

def start_web_interface(nodes):
    st.title("ChronaNet Dashboard")
    st.header("Estat de la xarxa")
    for node in nodes:
        st.subheader(f"Node {node.node_id}")
        st.write(f"Estat: {'Actiu' if node.is_active else 'Inactiu'}")
        st.write(f"Saldo: {node.wallet.balance} Chronacoins")
        st.write(f"Nombre de blocs: {len(node.blockchain.chain)}")
        if node.blockchain.chain:
            st.write("Últim bloc:")
            st.json(node.blockchain.chain[-1])
        st.write("Historial de transaccions:")
        st.json(node.wallet.history)
        t = datetime.now().timestamp()
        coherence = node.coherence(t, nodes)
        st.write(f"Coherència actual: {coherence:.3f}")
        st.line_chart([coherence])

def simulate_network(n_nodes=3, t=0.12345):
    tesseracts = [
        [Tesseract(i+1, (2+i%3, 3+i%4, 5+i%5), i*0.1, 1.0 + i*0.01) for i in range(16)]
        for _ in range(n_nodes)
    ]
    ports = [5000 + i for i in range(n_nodes)]
    nodes = [ChronaNetNode(f"Node_{j+1}", tess, port=ports[j]) for j, tess in enumerate(tesseracts)]

    for i, node in enumerate(nodes):
        node.add_peer(f"http://localhost:{ports[(i+1)%n_nodes]}")

    for node in nodes:
        start_node_server(node)

    for node in nodes:
        if not node.sync_phase(t, nodes):
            continue

    for node in nodes:
        node.wallet.update_balance(1000.0, "initial", {"note": "Saldo inicial per a la simulació"})

    transactions = [
        nodes[0].create_transaction("Node_2", 100, t, "Hg-Ax-Ro"),
        nodes[1].create_transaction("Node_3", 50, t + 0.1, "Li-St-Tx"),
        nodes[2].create_transaction("Node_1", 75, t + 0.2, "Hg-Ax-Ro"),
        nodes[0].create_transaction("Node_2", 25, t + 0.3, "Li-St-Tx")
    ]

    for txn in transactions:
        if txn:
            for node in nodes:
                if node.is_active:
                    node.validate_transaction(txn, txn['timestamp'], nodes)

    start_web_interface(nodes)

if __name__ == "__main__":
=======
"""
ChronaNet Framework
Copyright (c) 2025 Luis Morató de Dalmases

This software is licensed under the MIT License for non-commercial use.
Commercial use requires a separate license: contact morato.lluis@gmail.com.
See LICENSE.md for full details: https://github.com/ChronaNet

Attribution:
ChronaCoin Protocol, developed by Luis Morató de Dalmases (2025).
Powered by quantum-temporal cryptography and chrono-algorithmic architecture.
"""

import numpy as np
import json
from scipy.fft import fft
import logging
from datetime import datetime
import hashlib
from flask import Flask, request, jsonify
import requests
import threading
import streamlit as st
import time

# Configuració de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TAU = 0.37037  # Període temporal base (segons)
COHERENCE_THRESHOLD = 0.707  # Llindar de coherència per PoC
PHASE_TOLERANCE = 1e-2  # Tolerància de fase (radians)
REWARD_BASE = 10.0  # Recompensa base en Chronacoins per bloc validat

class Tesseract:
    def __init__(self, id, triad, phase, intensity):
        self.id = id
        self.triad = triad
        self.phase = phase
        self.intensity = intensity
        self.omega = self._compute_omega()

    def _compute_omega(self):
        rad_val = self._radical(*self.triad)
        return 2 * np.pi / (TAU / np.log(rad_val) if rad_val > 1 else TAU)

    def _radical(self, a, b, c):
        def prime_factors(n):
            factors = set()
            d = 2
            while n > 1:
                if n % d == 0:
                    factors.add(d)
                    n //= d
                d += 1
                if d * d > n and n > 1:
                    factors.add(n)
                    break
            return factors
        return np.prod(list(prime_factors(a) | prime_factors(b) | prime_factors(c)))

    def psi(self, t):
        return self.intensity * np.exp(1j * (self.omega * t + self.phase))

class CronaWallet:
    def __init__(self, tesseract_t8, wallet_file=None):
        self.t8 = tesseract_t8
        self.phase_key = self._compute_phase_key()
        self.wallet_file = wallet_file or f"wallet_{tesseract_t8.id}.json"
        self.balance = 0.0
        self.history = []
        self.load_wallet()

    def _compute_phase_key(self, t_max=1.0, dt=0.01):
        t_vec = np.arange(0, t_max, dt)
        psi_vec = [self.t8.psi(t) for t in t_vec]
        return np.trapz(psi_vec, t_vec)

    def sign_transaction(self, txn, t):
        txn['phase_key'] = abs(self.phase_key)
        return txn

    def update_balance(self, amount, txn_type, txn):
        self.balance += amount
        self.history.append({
            "type": txn_type,
            "amount": amount,
            "txn": txn,
            "timestamp": datetime.now().timestamp()
        })
        self.save_wallet()

    def save_wallet(self):
        try:
            with open(self.wallet_file, 'w') as f:
                json.dump({"balance": self.balance, "history": self.history}, f, indent=2)
            logger.info(f"Wallet guardat a {self.wallet_file}")
        except IOError as e:
            logger.error(f"Error guardant wallet: {e}")

    def load_wallet(self):
        try:
            with open(self.wallet_file, 'r') as f:
                data = json.load(f)
                self.balance = data.get("balance", 0.0)
                self.history = data.get("history", [])
            logger.info(f"Wallet carregat des de {self.wallet_file}")
        except FileNotFoundError:
            logger.info(f"No s'ha trobat {self.wallet_file}. Iniciant wallet buit.")

class ChronaBlockChain:
    def __init__(self, node_id):
        self.chain = []
        self.current_transactions = []
        self.node_id = node_id
        self.ledger_path = f"ledger_examples/blockchain_{node_id}.json"
        self.load_blockchain()

    def add_transaction(self, txn):
        self.current_transactions.append(txn)

    def create_block(self, t, coherence, t8_hash, prev_hash="0"):
        block = {
            "block_id": len(self.chain) + 1,
            "timestamp": t,
            "transactions": self.current_transactions,
            "coherence": coherence,
            "t8_hash": list(np.abs(t8_hash[:10])),
            "prev_hash": prev_hash
        }
        block["block_hash"] = hashlib.sha256(json.dumps(block, sort_keys=True).encode()).hexdigest()
        self.chain.append(block)
        self.current_transactions = []
        self.save_blockchain()
        return block

    def save_blockchain(self):
        try:
            with open(self.ledger_path, 'w') as f:
                json.dump(self.chain, f, indent=2)
            logger.info(f"Blockchain guardat a {self.ledger_path}")
        except IOError as e:
            logger.error(f"Error guardant blockchain: {e}")

    def load_blockchain(self):
        try:
            with open(self.ledger_path, 'r') as f:
                self.chain = json.load(f)
            logger.info(f"Blockchain carregat des de {self.ledger_path}")
        except FileNotFoundError:
            logger.info(f"No s'ha trobat {self.ledger_path}. Iniciant blockchain buit.")

    def validate_block(self, block):
        expected_hash = hashlib.sha256(json.dumps({k: block[k] for k in block if k != 'block_hash'}, sort_keys=True).encode()).hexdigest()
        return block["block_hash"] == expected_hash and block["coherence"] >= COHERENCE_THRESHOLD

class ChronaNetNode:
    def __init__(self, node_id, tesseracts, port, tavari_db='tavari_symbols.json'):
        self.node_id = node_id
        self.tesseracts = tesseracts
        self.t8 = next(tess for tess in tesseracts if tess.id == 8)
        self.wallet = CronaWallet(self.t8, f"ledger_examples/wallet_{node_id}.json")
        self.tavari_db = self._load_tavari_db(tavari_db)
        self.phase_vector = None
        self.is_active = True
        self.blockchain = ChronaBlockChain(node_id)
        self.ledger_path = f"ledger_examples/ledger_{node_id}.json"
        self.port = port
        self.peers = []
        logger.info(f"Node {node_id} inicialitzat amb {len(tesseracts)} tesseracts al port {port}.")
        self.sync_chain()

    def _load_tavari_db(self, tavari_file):
        default_db = {
            "Hg-Ax-Ro": {"value": 123, "freq_mod": 7.83},
            "Li-St-Tx": {"value": 456, "freq_mod": 11.7}
        }
        try:
            with open(tavari_file, 'w') as f:
                json.dump(default_db, f)
            with open(tavari_file, 'r') as f:
                return json.load(f)
        except IOError:
            logger.warning("Error d'E/S amb tavari_symbols.json. Usant base de dades per defecte.")
            return default_db

    def save_transaction(self, txn, valid=True):
        record = txn.copy()
        record["valid"] = valid
        try:
            with open(self.ledger_path, 'r') as f:
                ledger = json.load(f)
        except FileNotFoundError:
            ledger = []
        ledger.append(record)
        with open(self.ledger_path, 'w') as f:
            json.dump(ledger, f, indent=2)

    def resonance_field(self, t):
        R = 0
        for i, tess_i in enumerate(self.tesseracts):
            for j, tess_j in enumerate(self.tesseracts[i+1:], i+1):
                chi_ij = np.log(tess_i._radical(*tess_i.triad)) / np.linalg.norm(
                    np.array(tess_i.triad) - np.array(tess_j.triad)
                )
                delta_phi = tess_i.phase - tess_j.phase + (tess_i.omega - tess_j.omega) * t
                R += chi_ij * np.cos(delta_phi)
        return R

    def spectral_hash(self, t, dt=0.01):
        t_vec = np.arange(0, t, dt)
        psi_vec = [self.t8.psi(ti) for ti in t_vec]
        return fft(psi_vec)

    def coherence(self, t, network_nodes):
        psi_sum = sum(node.t8.psi(t) for node in network_nodes if node.is_active)
        norm_sum = sum(abs(node.t8.psi(t)) for node in network_nodes if node.is_active)
        C = abs(psi_sum) / norm_sum if norm_sum != 0 else 0
        return C

    def sync_phase(self, t, network_nodes):
        self.phase_vector = np.array([np.cos(self.t8.phase + self.t8.omega * t),
                                      np.sin(self.t8.phase + self.t8.omega * t)])
        for node in network_nodes:
            if node.node_id != self.node_id and node.is_active:
                other_vector = node.phase_vector
                if other_vector is not None:
                    dot_product = np.dot(self.phase_vector, other_vector)
                    if dot_product < np.cos(PHASE_TOLERANCE):
                        self.is_active = False
                        logger.warning(f"Node {self.node_id} desactivat per desincronització de fase.")
                        return False
        return True

    def map_tavari(self, symbol):
        omega = self.t8.omega
        tavari = self.tavari_db.get(symbol, {"value": sum(ord(c) for c in ''.join(symbol))})
        return tavari["value"] % int(omega * 100)

    def encode_message(self, msg, t, tavari):
        R = self.resonance_field(t)
        tavari_mod = self.map_tavari(tavari)
        return ''.join(chr((ord(c) ^ int(t * R)) + tavari_mod) for c in msg)

    def decode_message(self, encoded, t, tavari):
        R = self.resonance_field(t)
        tavari_mod = self.map_tavari(tavari)
        return ''.join(chr(((ord(c) - tavari_mod) ^ int(t * R)) % 256) for c in encoded)  # Correcció i assegurar valors vàlids

    def create_transaction(self, recipient_id, value, t, tavari):
        if self.wallet.balance < value:
            logger.error(f"Saldo insuficient a {self.node_id}: {self.wallet.balance} < {value}")
            return None
        txn = {
            'sender': self.node_id,
            'recipient': recipient_id,
            'triad': self.t8.triad,
            'phi': self.t8.phase,
            'omega': self.t8.omega,
            'value': value,
            'timestamp': t,
            'tavari': tavari,
            'hash': self.spectral_hash(t).tolist()
        }
        txn = self.wallet.sign_transaction(txn, t)
        self.wallet.update_balance(-value, "sent", txn)
        logger.info(f"Transacció creada: {txn['sender']} -> {txn['recipient']}, valor={value}")
        return txn

    def validate_transaction(self, txn, t, network_nodes):
        if not self.is_active:
            logger.error(f"Node {self.node_id} inactiu. Validació fallida.")
            return False
        C = self.coherence(t, network_nodes)
        if C < COHERENCE_THRESHOLD:
            logger.warning(f"Coherència insuficient: C={C:.3f} < {COHERENCE_THRESHOLD}")
            return False
        expected_hash = self.spectral_hash(t)
        hash_diff = abs(np.array(txn['hash']) - expected_hash).sum()
        if hash_diff > 1e-3:
            logger.error(f"Hash espectral no coincident: diff={hash_diff}")
            return False
        phase_key_valid = abs(txn['phase_key'] - abs(self.wallet.phase_key)) < 1e-2
        if not phase_key_valid:
            logger.error("Clau de fase invàlida.")
            return False
        logger.info(f"Transacció validada: {txn['sender']} -> {txn['recipient']}")
        self.save_transaction(txn, valid=True)
        self.blockchain.add_transaction(txn)
        if txn['recipient'] == self.node_id:
            self.wallet.update_balance(txn['value'], "received", txn)
        if len(self.blockchain.current_transactions) >= 3:
            prev_hash = self.blockchain.chain[-1]["block_hash"] if self.blockchain.chain else "0"
            block = self.blockchain.create_block(t, C, self.spectral_hash(t), prev_hash)
            reward = REWARD_BASE * C
            self.wallet.update_balance(reward, "reward", {"block_id": block["block_id"]})
            logger.info(f"Bloc creat per node {self.node_id}: {block['block_id']} amb {len(block['transactions'])} transaccions. Recompensa: {reward} Chronacoins.")
            self.propagate_block(block)
        return True

    def add_peer(self, peer_url):
        if peer_url not in self.peers:
            self.peers.append(peer_url)
            logger.info(f"Peer afegit a {self.node_id}: {peer_url}")

    def sync_chain(self):
        longest_chain = self.blockchain.chain
        max_length = len(longest_chain)
        max_coherence = sum(block["coherence"] for block in longest_chain) if longest_chain else 0
        for peer in self.peers:
            try:
                response = requests.get(f"{peer}/chain")
                if response.status_code == 200:
                    peer_chain = response.json()
                    peer_length = len(peer_chain)
                    peer_coherence = sum(block["coherence"] for block in peer_chain) if peer_chain else 0
                    if peer_length > max_length or (peer_length == max_length and peer_coherence > max_coherence):
                        if all(self.blockchain.validate_block(block) for block in peer_chain):
                            longest_chain = peer_chain
                            max_length = peer_length
                            max_coherence = peer_coherence
            except requests.RequestException as e:
                logger.error(f"Error sincronitzant amb {peer}: {e}")
        if longest_chain != self.blockchain.chain:
            self.blockchain.chain = longest_chain
            self.blockchain.save_blockchain()
            logger.info(f"Blockchain sincronitzat a {self.node_id} amb {len(longest_chain)} blocs.")

    def propagate_block(self, block):
        for peer in self.peers:
            try:
                response = requests.post(f"{peer}/block", json=block)
                if response.status_code == 200:
                    logger.info(f"Bloc propagat a {peer}")
                else:
                    logger.warning(f"Error propagant bloc a {peer}: {response.status_code}")
            except requests.RequestException as e:
                logger.error(f"Error de connexió amb {peer}: {e}")

    def receive_block(self, block):
        if self.blockchain.validate_block(block):
            self.blockchain.chain.append(block)
            self.blockchain.save_blockchain()
            logger.info(f"Bloc rebut i afegit a {self.node_id}: {block['block_id']}")
        else:
            logger.warning(f"Bloc rebut invàlid a {self.node_id}")

def start_node_server(node, host='localhost'):
    app = Flask(__name__)

    @app.route('/transaction', methods=['POST'])
    def receive_transaction():
        txn = request.get_json()
        node.validate_transaction(txn, txn['timestamp'], [node])
        return jsonify({"status": "Transaction received"}), 200

    @app.route('/block', methods=['POST'])
    def receive_block():
        block = request.get_json()
        node.receive_block(block)
        return jsonify({"status": "Block received"}), 200

    @app.route('/chain', methods=['GET'])
    def get_chain():
        return jsonify(node.blockchain.chain), 200

    threading.Thread(target=app.run, kwargs={'host': host, 'port': node.port}).start()

def start_web_interface(nodes):
    st.title("ChronaNet Dashboard")
    st.header("Estat de la xarxa")
    for node in nodes:
        st.subheader(f"Node {node.node_id}")
        st.write(f"Estat: {'Actiu' if node.is_active else 'Inactiu'}")
        st.write(f"Saldo: {node.wallet.balance} Chronacoins")
        st.write(f"Nombre de blocs: {len(node.blockchain.chain)}")
        if node.blockchain.chain:
            st.write("Últim bloc:")
            st.json(node.blockchain.chain[-1])
        st.write("Historial de transaccions:")
        st.json(node.wallet.history)
        t = datetime.now().timestamp()
        coherence = node.coherence(t, nodes)
        st.write(f"Coherència actual: {coherence:.3f}")
        st.line_chart([coherence])

def simulate_network(n_nodes=3, t=0.12345):
    tesseracts = [
        [Tesseract(i+1, (2+i%3, 3+i%4, 5+i%5), i*0.1, 1.0 + i*0.01) for i in range(16)]
        for _ in range(n_nodes)
    ]
    ports = [5000 + i for i in range(n_nodes)]
    nodes = [ChronaNetNode(f"Node_{j+1}", tess, port=ports[j]) for j, tess in enumerate(tesseracts)]

    for i, node in enumerate(nodes):
        node.add_peer(f"http://localhost:{ports[(i+1)%n_nodes]}")

    for node in nodes:
        start_node_server(node)

    for node in nodes:
        if not node.sync_phase(t, nodes):
            continue

    for node in nodes:
        node.wallet.update_balance(1000.0, "initial", {"note": "Saldo inicial per a la simulació"})

    transactions = [
        nodes[0].create_transaction("Node_2", 100, t, "Hg-Ax-Ro"),
        nodes[1].create_transaction("Node_3", 50, t + 0.1, "Li-St-Tx"),
        nodes[2].create_transaction("Node_1", 75, t + 0.2, "Hg-Ax-Ro"),
        nodes[0].create_transaction("Node_2", 25, t + 0.3, "Li-St-Tx")
    ]

    for txn in transactions:
        if txn:
            for node in nodes:
                if node.is_active:
                    node.validate_transaction(txn, txn['timestamp'], nodes)

    start_web_interface(nodes)

if __name__ == "__main__":
>>>>>>> d323bb4b1003c68d8d6703c5859b8de4e111a411
    simulate_network()