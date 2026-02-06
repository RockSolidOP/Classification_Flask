(() => {
  const busy = document.createElement('div');
  busy.className = 'ut-busy-overlay';
  busy.innerHTML = '<div class="ut-busy-box"><div class="ut-spinner"></div><div id="busyMsg">Working...</div></div>';
  document.body.appendChild(busy);

  function setBusy(visible, msg) {
    const msgEl = document.getElementById('busyMsg');
    if (msgEl && msg) msgEl.textContent = msg;
    busy.style.display = visible ? 'flex' : 'none';
  }

  async function getSuggestedVectorVersion(url) {
    try {
      const res = await fetch(url);
      const data = await res.json();
      if (data.ok && data.version) return data.version;
    } catch (e) {}
    return 'v1';
  }

  async function buildVectors(url, version) {
    setBusy(true, 'Building vectors...');
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ version })
      });
      const data = await res.json();
      if (!data.ok) return alert('Vector build failed: ' + (data.error || ''));
      alert('Vectors built and activated: ' + data.version);
      location.reload();
    } catch (e) {
      alert('Vector build error');
    } finally {
      setBusy(false);
    }
  }

  async function buildVectorsManual(urlVersion, urlBuild) {
    const suggested = await getSuggestedVectorVersion(urlVersion);
    const version = prompt('Enter version name for new vectors (Cancel to skip)', suggested);
    if (version === null) return;
    const trimmed = version.trim();
    if (!trimmed) return;
    await buildVectors(urlBuild, trimmed);
  }

  async function deleteAll(url) {
    if (!confirm('Delete ALL curated records and images? This cannot be undone.')) return;
    setBusy(true, 'Deleting all curated records...');
    try {
      const res = await fetch(url, { method: 'POST' });
      const data = await res.json();
      if (!data.ok) {
        alert('Delete all failed');
      } else {
        const featureImgs = data.images_deleted ?? 0;
        alert(`Deleted ${data.deleted} records and removed ${featureImgs} images`);
        location.reload();
      }
    } catch (e) {
      alert('Delete all error');
    } finally {
      setBusy(false);
    }
  }

  async function deleteLabel(url, label) {
    if (!confirm(`Delete all curated records for label ${label}?`)) return;
    setBusy(true, 'Deleting label records...');
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label, delete_images: true })
      });
      const data = await res.json();
      if (!data.ok) {
        alert('Delete label failed: ' + (data.error || res.status));
      } else {
        const removed = data.deleted ?? 0;
        alert(`Deleted ${removed} records for ${label}.`);
        location.reload();
      }
    } catch (e) {
      alert('Delete label error');
    } finally {
      setBusy(false);
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    const urlsEl = document.getElementById('curatedUrls');
    if (!urlsEl) return;
    const urls = JSON.parse(urlsEl.textContent || '{}');

    const buildBtn = document.querySelector('[data-build-vectors]');
    buildBtn?.addEventListener('click', () => buildVectorsManual(urls.vector_version, urls.build_vectors));

    const deleteAllBtn = document.querySelector('[data-delete-all]');
    deleteAllBtn?.addEventListener('click', () => deleteAll(urls.delete_all));

    document.querySelectorAll('[data-delete-label]').forEach((btn) => {
      const label = btn.getAttribute('data-delete-label');
      btn.addEventListener('click', () => deleteLabel(urls.delete_label, label));
    });
  });
})();

