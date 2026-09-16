# Networking background

This document covers the networking side of the project: what a network flow
is, how each attack in CIC-IDS2017 works at the protocol level, how those
attacks show up in the flow features we measured, what the model actually
relies on, and what it would take to deploy a flow-based intrusion detection
system (IDS) on a real network.

All numbers quoted here come from `04_feature_importance.py` (results in
`results/features/`). Times are in microseconds, as CICFlowMeter reports them;
sizes are in bytes.

---

## 1. What the data is: network flows

A **flow** is the set of packets exchanged between two endpoints, identified by
the 5-tuple (source IP, destination IP, source port, destination port,
protocol), from the first packet until the connection closes or a timeout
expires. CIC-IDS2017 was built by capturing every packet on a test network for
five days and passing the captures through **CICFlowMeter**, which groups
packets into **bidirectional** flows: the *forward* direction is the direction
of the first packet (normally client to server), the *backward* direction is
the reply.

For every flow, CICFlowMeter computes statistics rather than keeping the
packets themselves:

| Group | Examples | What it describes |
|---|---|---|
| Volume | Total Fwd Packets, Total Length of Bwd Packets | How much was sent each way |
| Packet sizes | Fwd Packet Length Mean, Max Packet Length | Small control packets vs large data packets |
| Timing | Flow Duration, Flow IAT Mean (inter-arrival time) | How fast and how regularly packets arrive |
| TCP flags | SYN / FIN / RST / PSH / URG Flag Count | How the connection was opened, used and closed |
| TCP setup | Init_Win_bytes_forward / backward | The receive window each side advertised at the start |
| Rates | Flow Bytes/s, Flow Packets/s | Throughput of the flow |
| Activity | Active Mean, Idle Mean | Bursts and pauses within the flow |

This is the same idea as the flow records that routers and switches already
export with **NetFlow** (RFC 3954) or its standard successor **IPFIX**
(RFC 7011), only with many more statistics per flow. A flow-based IDS
therefore never needs to read packet payloads, which has two consequences:

- It keeps working when traffic is encrypted (HTTPS, SSH), because timing,
  sizes and flags are still visible.
- It cannot see *what* was sent. Attacks that live entirely in the payload,
  such as SQL injection or cross-site scripting, look like ordinary web
  requests at the flow level. Section 3 shows this is exactly where our models
  are weakest.

## 2. Types of intrusion detection

| | Signature-based | Anomaly-based |
|---|---|---|
| How it decides | Matches known attack patterns (for example Snort or Suricata rules) | Flags traffic that deviates from a model of normal behaviour |
| Strength | Precise on known attacks, few false alarms | Can catch new attacks |
| Weakness | Blind to anything without a rule | More false alarms |

A **network** IDS (NIDS) watches traffic on the wire; a **host** IDS watches
logs and processes on one machine. This project builds a *supervised,
flow-based NIDS*: the classifiers learn each attack class from labelled
examples. In practice that puts it closer to signature-based detection (it
recognises the attack types it was trained on) but with learned rather than
hand-written patterns.

## 3. The attacks, and how they show up in the flows

For each attack, the table gives the mechanism and the features that on their
own best separate it from benign traffic. *Separation* runs from 0 (the
feature alone cannot tell attack from benign) to 1 (it separates them
perfectly); it is `|AUC - 0.5| x 2` for that single feature.

### Brute force: FTP-Patator and SSH-Patator

Patator tries username and password combinations against a login service,
over and over. FTP uses TCP port 21 and SSH TCP port 22.

| Feature | Attack (median) | Benign (median) | Separation |
|---|---|---|---|
| Destination Port (FTP-Patator) | 21 | 80 | 1.00 |
| Destination Port (SSH-Patator) | 22 | 80 | 0.99 |
| Total Backward Packets (SSH) | 32 | 2 | 0.84 |

Each SSH attempt runs a full key exchange and authentication dialogue, which
is why its flows carry 16 times more reply packets than a typical benign flow.

### Slow denial of service: slowloris and Slowhttptest

A web server can only hold so many connections open. These tools open many
connections and keep each one alive by sending an HTTP request *very slowly*
(a few header bytes every few seconds), so the server waits and eventually
runs out of connection slots. Almost no bandwidth is needed.

| Feature | slowloris | Slowhttptest | Benign |
|---|---|---|---|
| Flow IAT Mean (gap between packets) | 5.9 s | 10.5 s | 0.016 s |
| Flow Packets/s | 0.2 | 0.1 | 85.9 |

The signature is exactly the mechanism: packets hundreds of times further
apart than normal.

### Volumetric HTTP denial of service: Hulk, GoldenEye and DDoS

**Hulk** floods a web server with HTTP requests whose URLs and parameters are
randomised, so caches cannot absorb them. **GoldenEye** holds many keep-alive
connections and keeps requests flowing through them. The **DDoS** in this
dataset is an HTTP flood from several attacking machines at once (LOIC).

| Attack | Feature | Attack (median) | Benign (median) | Separation |
|---|---|---|---|---|
| DoS Hulk | Bwd Packet Length Std | 2,122 | 0 | 0.87 |
| DoS Hulk | Fwd IAT Max | 85.9 s | 4 us | 0.86 |
| GoldenEye | Flow Packets/s | 1.0 | 85.9 | 0.76 |
| GoldenEye | Flow IAT Mean | 1.1 s | 0.016 s | 0.76 |
| DDoS | Flow IAT Std | 0.96 s | 134 us | 0.64 |

Hulk's randomised requests produce responses of widely varying size (a large
spread in reply packet length), over long-lived connections.

### Heartbleed (CVE-2014-0160)

A bug in OpenSSL's TLS *heartbeat* extension: the client sends a heartbeat
message claiming a payload longer than it really is, and a vulnerable server
replies with that many bytes copied from its own memory, up to 64 KB per
message. The leak travels in the server's replies.

| Feature | Heartbleed (median) | Benign (median) | Separation |
|---|---|---|---|
| Bwd Packet Length Max | 14,480 | 90 | 1.00 |
| Bwd Packet Length Mean | 3,773 | 83 | 1.00 |

Enormous reply packets separate all 11 Heartbleed flows perfectly, which is
why even a model that saw only 7 of them in training can find them.

### PortScan

A port scanner probes many ports on a target to find open services. Each probe
is a tiny flow: a connection attempt and a one-packet answer (SYN-ACK for an
open port, RST for a closed one), with no data.

| Feature | PortScan (median) | Benign (median) | Separation |
|---|---|---|---|
| Bwd Packets/s | 20,000 | 26.3 | 0.86 |
| Total Length of Fwd Packets | 0 | 68 | 0.81 |

Zero forward payload is the probe itself. The huge packets-per-second value is
an artefact of very short flows: one reply packet within about 50
microseconds works out to 20,000 packets per second.

### Bot (Ares)

Infected machines repeatedly contact a command-and-control server to fetch
instructions.

| Feature | Bot (median) | Benign (median) | Separation |
|---|---|---|---|
| Destination Port | 8,080 | 80 | 0.65 |
| Average Packet Size | 9 | 77 | 0.47 |

Short, near-empty polling messages to the command server's port.

### Infiltration

After a victim inside the network opens a malicious file, the attacker uses
that machine as a foothold and scans the internal network from it.

| Feature | Infiltration (median) | Benign (median) | Separation |
|---|---|---|---|
| Fwd IAT Total | 83.1 s | 4 us | 0.81 |
| Total Fwd Packets | 26 | 2 | 0.80 |

Long flows with many outgoing packets. Only 36 such flows exist, and they come
from a legitimate internal machine, which is what makes infiltration hard to
catch.

### Web attacks: brute force, XSS and SQL injection

All three are HTTP requests to a deliberately vulnerable web application
(DVWA): repeated login attempts, script injection into pages, and SQL fragments
in form fields. The attack is in the *content* of the requests, which flow
features cannot see.

| Attack | Feature | Attack (median) | Benign (median) | Separation |
|---|---|---|---|---|
| Web brute force | Init_Win_bytes_backward | 28,960 | -1 | 0.85 |
| Web brute force | Flow Bytes/s | 0 | 4,459.5 | 0.82 |
| XSS | Init_Win_bytes_backward | 28,960 | -1 | 0.85 |
| XSS | Flow Bytes/s | 0 | 4,459.5 | 0.85 |
| SQL injection | Fwd Packet Length Min | 0 | 6 | 0.60 |

Two findings stand out:

- **Brute force and XSS have the same signature.** Their strongest features
  and median values are identical, so a model has almost nothing to tell them
  apart with. This is why XSS is the class the models most often confuse
  (F1 0.45 for the decision tree), usually with web brute force.
- **The typical web-attack flow carries no payload.** A median of 0 bytes per
  second, and a largest packet of 0 bytes for XSS, means most of these flows
  contain only TCP control packets. This matches the flow-construction problem
  reported by Engelen et al. (2021): after a connection closes, CICFlowMeter
  can emit its last packets as a separate flow, which inherits the attack
  label although it contains no attack.

(An initial window of -1 means CICFlowMeter saw no TCP window in that
direction, as with the many UDP flows, such as DNS, in benign traffic.)

## 4. What the model actually relies on

Permutation importance measures how much macro F1 drops when one feature's
values are shuffled, which breaks its link to the label. The top three
features for the Random Forest are:

| Rank | Feature | Drop in macro F1 |
|---|---|---|
| 1 | Destination Port | 0.021 |
| 2 | Init_Win_bytes_forward | 0.020 |
| 3 | Init_Win_bytes_backward | 0.014 |

All three are **shortcuts** more than attack behaviour:

- **Destination port** identifies the *service*, not the attack. "Port 21
  means FTP-Patator" holds in this dataset because the only FTP traffic in the
  capture was the attack. On a real network full of legitimate FTP and SSH,
  or with services moved to other ports, this rule breaks.
- **The initial TCP window** is chosen by the operating system's TCP stack.
  29,200, the forward window of the web attacks, is a common Linux default.
  In a small test network where attacks come from a few Linux attack machines
  and most benign traffic comes from Windows clients, the window size
  fingerprints *which machine* sent the traffic rather than *what* it did.
  A model trained here would learn "traffic from the attack machines" and
  would not transfer to a network with different machines.

This does not make the results wrong, but it means the high scores on
CIC-IDS2017 overstate how well these models would do elsewhere. Training
without these features, and testing on data from a different network, are the
natural next checks.

## 5. Data quality issues found

| Issue | Measured | Consequence |
|---|---|---|
| Exact duplicate rows | 256,684 rows (9.07%) | Copies land in both training and test data, so part of the test set is memorised; removed before training. Measured effect on accuracy is small (+0.0002), but macro F1 is distorted |
| Constant columns | 8 | Carry no information; removed |
| Duplicated column | `Fwd Header Length` appears twice | Removed |
| Identical feature pairs | 8 pairs, for example `Total Fwd Packets == Subflow Fwd Packets` | The 69 features hold only 61 distinct measurements |
| Implausible identities | `Fwd PSH Flags == SYN Flag Count` on every flow | Two different TCP flags cannot always match; points to a bug in how the tool computed them |

Engelen et al. (2021) examined CIC-IDS2017 in detail and report further
problems, including mislabelled flows and flow-construction errors, so results
on this dataset should be read with those limits in mind.

## 6. Deploying a flow-based IDS

A flow-based IDS on a real network is a pipeline:

```
 traffic --> mirror port / TAP --> flow meter --> collector --> classifier --> alert
             (copy of packets)     (CICFlowMeter,   (stores     (the trained
                                    or NetFlow/IPFIX flow        model)
                                    from routers)    records)
```

Points that follow from how flows work:

- **It runs beside the traffic, not in its path.** A flow's statistics are only
  complete when the flow ends, so the classifier sees traffic after the fact.
  It can raise an alert or trigger a firewall rule, but it cannot block the
  first packets of an attack the way an inline intrusion *prevention* system
  can.
- **Detection delay depends on the attack.** A port-scan probe lasts
  microseconds and is classified almost at once. A slowloris connection is
  designed to stay open, so its flow, and the verdict on it, may not finish
  until the flow meter's timeout expires. Shorter timeouts give faster
  verdicts but compute features on partial flows.
- **Speed matters as much as accuracy.** A busy link produces a steady stream
  of flow records, and the classifier must keep up. `02_experiment.py`
  measures prediction throughput for every model (flows per second), shown in
  `results/charts/06_throughput.png`.
- **Encryption is not a problem; payload attacks are.** Timing, size and flag
  features survive encryption, which suits DoS, scans and brute force. Attacks
  hidden in payloads (SQL injection, XSS) need payload inspection or
  application logs as well.

## References

- Sharafaldin, I., Habibi Lashkari, A. and Ghorbani, A. A. (2018). *Toward
  Generating a New Intrusion Detection Dataset and Intrusion Traffic
  Characterization.* Proceedings of ICISSP 2018. (The CIC-IDS2017 dataset.)
- Engelen, G., Rimmer, V. and Joosen, W. (2021). *Troubleshooting an Intrusion
  Detection Dataset: the CICIDS2017 Case Study.* IEEE Security and Privacy
  Workshops (SPW).
- Claise, B. (2004). *Cisco Systems NetFlow Services Export Version 9.* RFC 3954.
- Claise, B., Trammell, B. and Aitken, P. (2013). *Specification of the IP Flow
  Information Export (IPFIX) Protocol.* RFC 7011.
- CVE-2014-0160 (Heartbleed), OpenSSL TLS heartbeat read overrun.
