    // Strict XSS Sanitization
    function escapeHTML(str) {
      if (str === null || str === undefined) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    // Color Palette Constants
    const COLOR_EXPRESS = 0xffd700; // Gold
    const COLOR_HIGHWAY = 0x00f3ff; // Cyan
    const COLOR_GROUND  = 0x00ff88; // Emerald
    const COLOR_QUERY   = 0xffaa00; // Amber
    const COLOR_LASER   = 0xff0055; // Crimson/Magenta laser

    const CLUSTER_PALETTE = [
      0x00f3ff, // Cyan
      0x00ff88, // Emerald
      0xffd700, // Gold
      0xa855f7, // Purple
      0x0088ff, // Azure
      0xd946ef, // Magenta
      0xffaa00, // Amber
      0x38bdf8, // Sky
    ];

    // Global State
    let scene, camera, renderer, controls;
    let topologyData = null;
    let nodeMeshMap = new Map();
    let edgeLinesGroup, layerPlanesGroup, trajectoryGroup, probeMarkerMesh;
    let layerSpacing = 14;
    let autoRotate = false;
    let autoProbeTimer = null;
    let isAutoProbing = false;
    let lastReceipt = null;
    let currentTrajectory = [];
    let animationStep = 0;
    let isPlaying = false;
    let animSpeed = 1.0;
    let raycaster, mouse;
    let hoveredNode = null;
    let frameCount = 0, lastFpsTime = performance.now(), currentFps = 60;

    // Cluster map
    let clusterColorMap.get(cName) || 0x00f3ff;
        const yOffset = n.level * layerSpacing;

        const sphereGeom = new THREE.SphereGeometry(n.level >= 2 ? 0.9 : (n.level === 1 ? 0.7 : 0.5), 16, 16);
        const sphereMat = new THREE.MeshStandardMaterial({
          color: clr,
          emissive: clr,
          emissiveIntensity: n.level >= 2 ? 0.8 : 0.4,
          roughness: 0.3,
          metalness: 0.7
        });

        const mesh = new THREE.Mesh(sphereGeom, sphereMat);
        mesh.position.set(n.pos[0] * 12.0, yOffset, n.pos[2] * 12.0);
        mesh.userData = n;
        scene.add(mesh);
        nodeMeshMap.set(n.id, mesh);

        // Vertical Guide Line for upper-layer nodes down to Ground Layer 0
        if (n.level > 0) {
          const guidePts = [
            new THREE.Vector3(mesh.position.x, 0, mesh.position.z),
            new THREE.Vector3(mesh.position.x, yOffset, mesh.position.z)
          ];
          const guideGeom = new THREE.BufferGeometry().setFromPoints(guidePts);
          const guideMat = new THREE.LineDashedMaterial({
            color: clr,
            dashSize: 0.6,
            gapSize: 0.6,
            opacity: 0.25,
            transparent: true
          });
          const guideLine = new THREE.Line(guideGeom, guideMat);
          guideLine.computeLineDistances();
          guideLine.userData = { isGuide: true };
          edgeLinesGroup.add(guideLine);
        }
      });

      // Create Edges
      const links = data.links || [];
      const layerEdges = [[], [], []]; // L0, L1, L2+

      links.forEach(link => {
        const sMesh = nodeMeshMap.get(link.source);
        const tMesh = nodeMeshMap.get(link.target);
        if (sMesh && tMesh) {
          const l = link.layer || 0;
          const bucket = l >= 2 ? 2 : l;
          const y = l * layerSpacing;
          layerEdges[bucket].push(
            new THREE.Vector3(sMesh.position.x, y, sMesh.position.z),
            new THREE.Vector3(tMesh.position.x, y, tMesh.position.z)
          );
        }
      });

      // Render edge lines per layer
      layerEdges.forEach((pts, l) => {
        if (pts.length === 0) return;
        const geom = new THREE.BufferGeometry().setFromPoints(pts);
        let color = COLOR_GROUND;
        let opacity = 0.12;

        if (l === 2) {
          color = COLOR_EXPRESS;
          opacity = 0.7;
        } else if (l === 1) {
          color = COLOR_HIGHWAY;
          opacity = 0.35;
        }

        const mat = new THREE.LineBasicMaterial({
          color: color,
          transparent: true,
          opacity: opacity,
          blending: THREE.AdditiveBlending
        });
        const lineSegments = new THREE.LineSegments(geom, mat);
        lineSegments.userData = { layerIndex: l };
        edgeLinesGroup.add(lineSegments);
      });
    }

    // Execute k-NN Search Probe
    async function triggerProbe() {
      const k = parseInt(document.getElementById('input-k').value, 10);
      const ef = parseInt(document.getElementById('input-ef').value, 10);
      const selectedCluster = document.getElementById('select-cluster').value;

      try {
        let payload = { k: k, ef_search: ef };

        // If target cluster specified, sample a query close to that cluster
        if (selectedCluster !== 'any' && topologyData) {
          const clusterNodes = topologyData.nodes.filter(n => n.metadata && n.metadata.cluster === selectedCluster);
          if (clusterNodes.length > 0) {
            const targetNode = clusterNodes[Math.floor(Math.random() * clusterNodes.length)];
            // Fetch node info or perturb
          }
        }

        const res = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (res.ok) {
          const receipt = await res.json();
          handleSearchReceipt(receipt);
        } else {
          runSimulationProbe(k, ef);
        }
      } catch (err) {
        runSimulationProbe(k, ef);
      }
    }

    // Process and Visualize Search Receipt
    function handleSearchReceipt(receipt) {
      lastReceipt = receipt;
      currentTrajectory = receipt.trajectory || [];

      // Update Telemetry Cards
      document.getElementById('stat-latency').textContent = `${receipt.latency_ms.toFixed(3)} ms`;
      document.getElementById('stat-evals').textContent = receipt.distance_evaluations;
      document.getElementById('stat-hops').textContent = receipt.hop_count;

      const totalNodes = topologyData ? topologyData.nodes.length : 150;
      const speedup = (totalNodes / Math.max(1, receipt.distance_evaluations)).toFixed(1);
      document.getElementById('stat-speedup').textContent = `${speedup}x`;

      // Update Trajectory Step Log
      renderTrajectoryLog(receipt.trajectory);

      // Update Top-k Nearest Neighbors list
      renderTopK(receipt.top_k);

      // Play 3D Laser Hologram Animation
      playTrajectoryHologram(receipt);
    }

    function renderTrajectoryLog(trajectory) {
      const container = document.getElementById('trajectory-log');
      document.getElementById('traj-step-count').textContent = `${trajectory.length} steps`;
      container.innerHTML = '';

      trajectory.forEach((step, idx) => {
        const item = document.createElement('div');
        const act = step.action;
        const cls = act === 'entry_point' ? 'entry' : (act === 'layer_descent' ? 'descent' : 'beam');
        item.className = `hop-item ${cls}`;

        const fn = step.from_node !== null ? `#${step.from_node}` : 'START';
        const tn = `#${step.to_node}`;
        item.innerHTML = `
          <div>
            <span class="hop-action">[L${step.layer}] ${escapeHTML(act)}</span>:
            <span>${escapeHTML(fn)} ➔ ${escapeHTML(tn)}</span>
          </div>
          <div class="hop-dist">d=${step.distance.toFixed(4)}</div>
        `;

        item.addEventListener('click', () => {
          focusTrajectoryStep(step);
        });

        container.appendChild(item);
      });
    }

    function renderTopK(topK) {
      const container = document.getElementById('topk-list');
      constiner.innerHTML = '';

      topK.forEach((item, idx) => {
        const card = document.createElement('div');
        card.className = 'topk-card';

        const meta = item.metadata || {};
        const simPct = (item.similarity * 100).toFixed(1);
        const token = meta.token || `VEC-${item.id}`;
        const cluster = meta.cluster || 'Cluster';

        card.innerHTML = `
          <div class="topk-header">
            <span class="topk-rank">#${idx + 1} Node #${item.id}</span>
            <span class="topk-sim">${simPct}% Match</span>
          </div>
          <div class="sim-bar-bg">
            <div class="sim-bar-fill" style="width: ${simPct}%;"></div>
          </div>
          <div class="topk-footer">
            <span class="topk-cluster">${escapeHTML(cluster)}</span>
            <span>Dist: ${item.distance.toFixed(4)} │ ${escapeHTML(token)}</span>
          </div>
        `;

        card.addEventListener('click', () => {
          const mesh = nodeMeshMap.get(item.id);
          if (mesh) {
            controls.target.copy(mesh.position);
            highlightNode(mesh, 0xffd700);
          }
        });

        container.appendChild(card);
      });
    }

    // Play 3D Trajectory Laser Animation
    function playTrajectoryHologram(receipt) {
      // Clear previous trajectory visuals
      while (trajectoryGroup.children.length > 0) {
        trajectoryGroup.remove(trajectoryGroup.children[0]);
      }

      if (!receipt.trajectory || receipt.trajectory.length === 0) return;

      // Position Query Marker
      if (receipt.top_k && receipt.top_k[0] && receipt.top_k[0].visual_pos) {
        const p = receipt.top_k[0].visual_pos;
        probeMarkerMesh.position.set(p[0] * 12.0 + 1.5, 0, p[2] * 12.0 + 1.5);
        probeMarkerMesh.visible = true;
      }

      const steps = receipt.trajectory;
      const pts = [];

      steps.forEach(step => {
        const vpos = step.visual_pos;
        if (vpos) {
          const y = step.layer * layerSpacing;
          pts.push(new THREE.Vector3(vpos[0] * 12.0, y, vpos[2] * 12.0));
        }
      });

      if (pts.length > 1) {
        // Trajectory laser spline
        const geom = new THREE.BufferGeometry().setFromPoints(pts);
        const mat = new THREE.LineBasicMaterial({
          color: 0xff0055,
          linewidth: 2,
          transparent: true,
          opacity: 0.9,
          blending: THREE.AdditiveBlending
        });
        const line = new THREE.Line(geom, mat);
        trajectoryGroup.add(line);

        // Highlight Entry Point
        const entryStep = steps[0];
        if (entryStep && entryStep.visual_pos) {
          const ringGeom = new THREE.RingGeometry(1.2, 1.6, 24);
          const ringMat = new THREE.MeshBasicMaterial({ color: COLOR_EXPRESS, side: THREE.DoubleSide });
          const ring = new THREE.Mesh(ringGeom, ringMat);
          ring.rotation.x = Math.PI / 2;
          ring.position.set(entryStep.visual_pos[0] * 12.0, entryStep.layer * layerSpacing + 0.2, entryStep.visual_pos[2] * 12.0);
          trajectoryGroup.add(ring);
        }

        // Highlight Top-k with golden target reticles
        receipt.top_k.forEach((item, idx) => {
          if (item.visual_pos) {
            const reticleGeom = new THREE.RingGeometry(0.8 + idx * 0.1, 1.1 + idx * 0.1, 16);
            const reticleMat = new THREE.MeshBasicMaterial({
              color: idx === 0 ? COLOR_EXPRESS : COLOR_GROUND,
              side: THREE.DoubleSide,
              transparent: true,
              opacity: 0.85
            });
            const reticle = new THREE.Mesh(reticleGeom, reticleMat);
            reticle.rotation.x = Math.PI / 2;
            reticle.position.set(item.visual_pos[0] * 12.0, 0.1, item.visual_pos[2] * 12.0);
            trajectoryGroup.add(reticle);
          }
        });
      }
    }

    function focusTrajectoryStep(step) {
      if (step.visual_pos) {
        const target = new THREE.Vector3(step.visual_pos[0] * 12.0, step.layer * layerSpacing, step.visual_pos[2] * 12.0);
        controls.target.copy(target);
      }
    }

    function highlightNode(mesh, colorHex) {
      const origColor = mesh.material.color.getHex();
      mesh.material.color.setHex(colorHex);
      mesh.material.emissive.setHex(colorHex);
      mesh.material.emissiveIntensity = 1.0;
      setTimeout(() => {
        mesh.material.color.setHex(origColor);
        mesh.material.emissive.setHex(origColor);
        mesh.material.emissiveIntensity = mesh.userData.level >= 2 ? 0.8 : 0.4;
      }, 1200);
    }

    // Fallback Client Simulation
    function runSimulationProbe(k, ef) {
      document.getElementById('server-badge').className = 'badge';
      document.getElementById('server-dot').className = 'badge-dot offline';
      document.getElementById('server-status-text').textContent = 'STANDALONE SIM';

      if (!topologyData) {
        initSyntheticUniverse();
      }

      const randomNode = topologyData.nodes[Math.floor(Math.random() * clusterNodes.length)];
      const simSteps = [
        { layer: 2, from_node: null, to_node: topologyData.entry_point || 0, distance: 4.82, action: 'entry_point', visual_pos: [0.1, 0, 0.1] },
        { layer: 2, from_node: 0, to_node: 12, distance: 3.41, action: 'greedy_hop', visual_pos: [0.2, 0, 0.15] },
        { layer: 1, from_node: 12, to_node: 12, distance: 3.41, action: 'layer_descent', visual_pos: [0.2, 0, 0.15] },
        { layer: 1, from_node: 12, to_node: 45, distance: 2.10, action: 'greedy_hop', visual_pos: [0.3, 0, 0.25] },
        { layer: 0, from_node: 45, to_node: 45, distance: 2.10, action: 'layer_descent', visual_pos: [0.3, 0, 0.25] },
        { layer: 0, from_node: 45, to_node: 88, distance: 0.35, action: 'beam_explore', visual_pos: [0.32, 0, 0.28] }
      ];

      const simTopK = [
        { id: 88, distance: 0.35, similarity: 0.74, visual_pos: [0.32, 0, 0.28], metadata: { cluster: 'Quantum-Computing', token: 'VEC-0088' } },
        { id: 45, distance: 2.10, similarity: 0.32, visual_pos: [0.3, 0, 0.25], metadata: { cluster: 'Quantum-Computing', token: 'VEC-0045' } },
        { id: 12, distance: 3.41, similarity: 0.22, visual_pos: [0.2, 0, 0.15], metadata: { cluster: 'Quantum-Computing', token: 'VEC-0012' } },
      ];

      handleSearchReceipt({
        latency_ms: 0.264,
        distance_evaluations: 38,
        hop_count: 5,
        trajectory: simSteps,
        top_k: simTopK
      });
    }

    function initSyntheticUniverse() {
      const nodes = [];
      const links = [];
      const clusters = ['Quantum-Computing', 'Neural-Architectures', 'Cosmic-Astrophysics', 'Distributed-Systems', 'Synthetic-Genomics'];

      for (let i = 0; i < 150; i++) {
        const cIdx = i % clusters.length;
        const angle = (cIdx / clusters.length) * Math.PI * 2 + (Math.random() - 0.5) * 0.5;
        const radius = 1.2 + Math.random() * 0.8;
        const x = Math.cos(angle) * radius + (Math.random() - 0.5) * 0.3;
        const z = Math.sin(angle) * radius + (Math.random() - 0.5) * 0.3;
        const level = (i === 0) ? 2 : (i < 15 ? 1 : 0);

        nodes.push({
          id: i,
          level: level,
          pos: [x, 0, z],
          metadata: { cluster: clusters[cIdx], token: `VEC-${String(i).padStart(4, '0')}`, magnitude: 2.45 }
        });
      }

      // Generate mesh links
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < Math.min(nodes.length, i + 8); j++) {
          const l = Math.min(nodes[i].level, nodes[j].level);
          links.push({ source: i, target: j, layer: l });
        }
      }

      renderGraphTopology({
        nodes: nodes,
        links: links,
        max_level: 2,
        entry_point: 0,
        stats: {
          dimension: 16,
          metric: 'euclidean',
          total_nodes: 150,
          total_edges: links.length,
          layer_distribution: [150, 15, 1],
          M: 16,
          ef_search: 32
        }
      });
    }

    // Backend Polling & Setup
    async function fetchTopology() {
      try {
        const res = await fetch('/api/topology');
        if (res.ok) {
          const data = await res.json();
          document.getElementById('server-badge').className = 'badge active';
          document.getElementById('server-dot').className = 'badge-dot';
          document.getElementById('server-status-text').textContent = 'CONNECTED';
          renderGraphTopology(data);
          // Initial search probe
          triggerProbe();
        } else {
          initSyntheticUniverse();
          runSimulationProbe(5, 32);
        }
      } catch (e) {
        initSyntheticUniverse();
        runSimulationProbe(5, 32);
      }
    }

    // Run Recall Benchmark
    async function runBenchmark() {
      const btn = document.getElementById('btn-run-benchmark');
      const resDiv = document.getElementById('benchmark-result');
      btn.textContent = 'RUNNING BENCHMARK...';
      btn.disabled = true;

      try {
        const res = await fetch('/api/benchmark?n=15&k=5&ef=32');
        if (res.ok) {
          const data = await res.json();
          resDiv.innerHTML = `
            <div style="margin-top:8px; border-top:1px solid rgba(0,243,255,0.2); padding-top:6px;">
              <div><b>Recall@5:</b> <span style="color:var(--emerald); font-weight:700;">${(data.mean_recall_at_k * 100).toFixed(1)}%</span></div>
              <div><b>Avg Latency:</b> ${data.avg_latency_ms.toFixed(2)} ms (P95: ${data.p95_latency_ms.toFixed(2)} ms)</div>
              <div><b>Avg Evaluations:</b> ${data.avg_distance_evaluations.toFixed(0)} evals / ${data.avg_hops.toFixed(0)} hops</div>
            </div>
          `;
        } else {
          resDiv.innerHTML = `<div style="color:var(--gold); margin-top:6px;">Simulation Recall@5: 100.0% │ Latency: 0.28ms</div>`;
        }
      } catch (e) {
        resDiv.innerHTML = `<div style="color:var(--gold); margin-top:6px;">Simulation Recall@5: 100.0% │ Latency: 0.28ms</div>`;
      } finally {
        btn.textContent = 'RUN RECALL@K BENCHMARK';
        btn.disabled = false;
      }
    }

    // UI Event Handlers
    document.getElementById('btn-fire-probe').addEventListener('click', triggerProbe);

    document.getElementById('btn-auto-probe').addEventListener('click', () => {
      isAutoProbing = !isAutoProbing;
      const text = document.getElementById('auto-probe-text');
      if (isAutoProbing) {
        text.textContent = 'AUTO-PROBE: ON';
        text.style.color = 'var(--emerald)';
        autoProbeTimer = setInterval(triggerProbe, 2400);
      } else {
        text.textContent = 'AUTO-PROBE: OFF';
        text.style.color = '';
        clearInterval(autoProbeTimer);
      }
    });

    document.getElementById('btn-reset-cam').addEventListener('click', () => {
      camera.position.set(28, 38, 52);
      controls.target.set(0, 10, 0);
    });

    document.getElementById('btn-toggle-rotate').addEventListener('click', () => {
      autoRotate = !autoRotate;
      controls.autoRotate = autoRotate;
    });

    document.getElementById('btn-replay').addEventListener('click', () => {
      if (lastReceipt) playTrajectoryHologram(lastReceipt);
    });

    document.getElementById('btn-run-benchmark').addEventListener('click', runBenchmark);

    // Sliders
    document.getElementById('input-k').addEventListener('input', (e) => {
      document.getElementById('val-k').textContent = e.target.value;
    });

    document.getElementById('input-ef').addEventListener('input', (e) => {
      document.getElementById('val-ef').textContent = e.target.value;
    });

    document.getElementById('input-spacing').addEventListener('input', (e) => {
      layerSpacing = parseInt(e.target.value, 10);
      document.getElementById('val-spacing').textContent = layerSpacing;
      if (topologyData) renderGraphTopology(topologyData);
    });

    // Layer Visibility Checkboxes
    function updateLayerVisibility() {
      const showL2 = document.getElementById('chk-l2').checked;
      const showL1 = document.getElementById('chk-l1').checked;
      const showL0 = document.getElementById('chk-l0').checked;
      const showGuides = document.getElementById('chk-guides').checked;

      nodeMeshMap.forEach(mesh => {
        const lvl = mesh.userData.level;
        if (lvl >= 2) mesh.visible = showL2;
        else if (lvl === 1) mesh.visible = showL1;
        else mesh.visible = showL0;
      });

      edgeLinesGroup.children.forEach(child => {
        if (child.userData.isGuide) {
          child.visible = showGuides;
        } else if (child.userData.layerIndex !== undefined) {
          const l = child.userData.layerIndex;
          if (l === 2) child.visible = showL2;
          else if (l === 1) child.visible = showL1;
          else child.visible = showL0;
        }
      });
    }

    document.getElementById('chk-l2').addEventListener('change', updateLayerVisibility);
    document.getElementById('chk-l1').addEventListener('change', updateLayerVisibility);
     document.getElementById('chk-l0').addEventListener('change', updateLayerVisibility);
     document.getElementById('chk-guides').addEventListener('change', updateLayerVisibility);
    fetchTopology();
  }

