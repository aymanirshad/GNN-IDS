"""Tiny fake NetFlow-v3-style CSV for dry-running the whole pipeline in minutes (NOT for results)."""
import sys, numpy as np, pandas as pd

n = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
rng = np.random.default_rng(0)
hosts = [f'10.0.{i // 250}.{i % 250}' for i in range(300)]
attackers = hosts[:6]
ts = np.sort(rng.integers(1_600_000_000_000, 1_600_000_000_000 + 6 * 3600 * 1000, n))
atk = rng.random(n) < 0.15
src = np.where(atk, rng.choice(attackers, n), rng.choice(hosts, n))
dst = np.where(atk, rng.choice(hosts[10:40], n), rng.choice(hosts, n))
f = {f'F{i}': rng.gamma(2.0, 50 * (i + 1), n) for i in range(12)}
for i in range(3):  # attacks shift a few features (weak signal so models do not hit 100%)
    f[f'F{i}'] = f[f'F{i}'] * np.where(atk, rng.uniform(0.6, 2.5, n), 1)
df = pd.DataFrame({'IPV4_SRC_ADDR': src, 'L4_SRC_PORT': rng.integers(1024, 65535, n), 'IPV4_DST_ADDR': dst, 'L4_DST_PORT': rng.choice([22, 80, 443, 53], n),
                   'PROTOCOL': rng.choice([6, 17], n), **f, 'FLOW_START_MILLISECONDS': ts, 'FLOW_END_MILLISECONDS': ts + rng.integers(1, 5000, n),
                   'Label': atk.astype(int), 'Attack': np.where(atk, rng.choice(['DoS', 'Scan', 'Exploits'], n), 'Benign')})
df.to_csv(sys.argv[1], index=False)
print('wrote', sys.argv[1], len(df))
