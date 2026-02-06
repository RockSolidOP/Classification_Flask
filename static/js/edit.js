(() => {
  function createBusyOverlay() {
    const existing = document.querySelector('.ut-busy-overlay');
    if (existing) return existing;
    const busy = document.createElement('div');
    busy.className = 'ut-busy-overlay';
    busy.innerHTML = '<div class="ut-busy-box"><div class="ut-spinner"></div><div id="busyMsg">Working...</div></div>';
    document.body.appendChild(busy);
    return busy;
  }

  const busy = createBusyOverlay();

  function setBusy(visible, msg) {
    const msgEl = document.getElementById('busyMsg');
    if (msgEl && msg) msgEl.textContent = msg;
    busy.style.display = visible ? 'flex' : 'none';
  }

  async function saveRow(page, urls) {
    setBusy(true, 'Saving...');
    const input = document.getElementById('label_' + page);
    const label = input ? input.value : '';
    try {
      const res = await fetch(urls.update(page), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label })
      });
      const data = await res.json();
      if (!data.ok) {
        alert('Save failed: ' + (data.error || res.status));
      } else if (input) {
        input.style.backgroundColor = '#e6fffa';
        setTimeout(() => { input.style.backgroundColor = ''; }, 500);
      }
    } catch (e) {
      alert('Save error');
    } finally {
      setBusy(false);
    }
  }

  async function restoreRow(page, urls) {
    setBusy(true, 'Restoring...');
    const input = document.getElementById('label_' + page);
    try {
      const res = await fetch(urls.restore(page), { method: 'POST' });
      const data = await res.json();
      if (!data.ok) {
        alert('Restore failed: ' + (data.error || res.status));
      } else if (input) {
        input.value = data.label;
        location.reload();
      }
    } catch (e) {
      alert('Restore error');
    } finally {
      setBusy(false);
    }
  }

  async function restoreAll(url) {
    setBusy(true, 'Restoring defaults...');
    try {
      const res = await fetch(url, { method: 'POST' });
      const data = await res.json();
      if (!data.ok) {
        alert('Restore all failed: ' + (data.error || res.status));
      } else {
        location.reload();
      }
    } catch (e) {
      alert('Restore all error');
    } finally {
      setBusy(false);
    }
  }

  function getSelectedPages() {
    return Array.from(document.querySelectorAll('.select-page:checked'))
      .map((cb) => parseInt(cb.value, 10))
      .filter((n) => Number.isFinite(n));
  }

  async function curateSelected(urls) {
    const pages = getSelectedPages();
    if (!pages.length) {
      alert('Select at least one page first.');
      return;
    }
    setBusy(true, 'Curating selected pages...');
    let success = 0;
    const failures = [];
    for (const page of pages) {
      try {
        const res = await fetch(urls.curate(page), { method: 'POST' });
        const data = await res.json();
        if (data.ok) {
          success += 1;
        } else {
          failures.push(page);
        }
      } catch (e) {
        failures.push(page);
      }
    }
    setBusy(false);
    if (!failures.length) {
      document.querySelectorAll('.select-page:checked').forEach((cb) => { cb.checked = false; });
    }
    let message = `Curated ${success} page${success === 1 ? '' : 's'}.`;
    if (failures.length) message += ` Failed: ${failures.join(', ')}`;
    alert(message);
  }

  async function curateAll(url) {
    setBusy(true, 'Curating all pages...');
    try {
      const res = await fetch(url, { method: 'POST' });
      const data = await res.json();
      if (!data.ok) {
        alert('Curate all failed: ' + (data.error || res.status));
      } else {
        alert('Curated ' + (data.count || 0) + ' pages');
      }
    } catch (e) {
      alert('Curate all error');
    } finally {
      setBusy(false);
    }
  }

  async function loadVectorVersions(url, selectId, noteId) {
    const select = document.getElementById(selectId);
    const note = document.getElementById(noteId);
    if (!select) return;
    select.disabled = true;
    try {
      const res = await fetch(url);
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || 'failed');
      const versions = data.versions || [];
      const active = data.active || 'default';
      select.innerHTML = '';
      if (!versions.length) {
        select.appendChild(new Option('default', 'default'));
      } else {
        versions.forEach((ver) => select.appendChild(new Option(ver, ver)));
      }
      if (active && !versions.includes(active)) {
        select.appendChild(new Option(`${active} (missing)`, active));
      }
      select.value = active;
      if (note) note.textContent = `Active: ${active}`;
    } catch (e) {
      select.innerHTML = '';
      select.appendChild(new Option('default', 'default'));
      alert('Could not load suggestion engines');
    } finally {
      select.disabled = false;
    }
  }

  async function setVectorVersion(url, version, selectId, noteId) {
    if (!version) return;
    const select = document.getElementById(selectId);
    const note = document.getElementById(noteId);
    setBusy(true, 'Switching suggestion engine...');
    try {
      const body = new URLSearchParams();
      body.append('version', version);
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: body.toString()
      });
      const data = await res.json();
      if (!data.ok) {
        alert('Failed to set suggestion engine: ' + (data.error || res.status));
      } else {
        if (note) note.textContent = `Active: ${version}`;
        alert('Suggestion engine set to ' + version);
      }
    } catch (e) {
      alert('Failed to set suggestion engine');
    } finally {
      setBusy(false);
      if (select) {
        await loadVectorVersions(url.replace('set_active_version', 'vector_versions'), selectId, noteId);
      }
    }
  }

  function attachRowHandlers(urls) {
    document.querySelectorAll('[data-save-page]').forEach((btn) => {
      const page = parseInt(btn.getAttribute('data-save-page'), 10);
      btn.addEventListener('click', () => saveRow(page, urls));
    });
    document.querySelectorAll('[data-restore-page]').forEach((btn) => {
      const page = parseInt(btn.getAttribute('data-restore-page'), 10);
      btn.addEventListener('click', () => restoreRow(page, urls));
    });
  }

  function setupSuggest(urls) {
    const overlay = document.getElementById('suggOverlay');
    const modal = document.getElementById('suggModal');
    if (!overlay || !modal) return;

    function openSuggest(page) {
      document.getElementById('suggPage').value = page;
      document.getElementById('suggList').innerHTML = '<li>Loading...</li>';
      document.getElementById('suggAliasInfo').textContent = '';
      overlay.style.display = 'block';
      modal.style.display = 'block';
      fetch(urls.suggest(page))
        .then(r => r.json()).then(data => {
          if (!data.ok) {
            document.getElementById('suggList').innerHTML = `<li>Error: ${data.error || 'not available'}</li>`;
            return;
          }
          const list = document.getElementById('suggList');
          list.innerHTML = '';
          (data.results || []).forEach((r, idx) => {
            const li = document.createElement('li');
            li.innerHTML = `<label><input type="radio" name="suggPick" value="${idx}" ${idx===0?'checked':''}/> ${r.label} <span style='color:#6b7280'>(score ${r.score.toFixed(3)})</span></label>`;
            list.appendChild(li);
          });
        }).catch(() => {
          document.getElementById('suggList').innerHTML = '<li>Error loading suggestions</li>';
        });
    }

    function closeSuggest() {
      overlay.style.display = 'none';
      modal.style.display = 'none';
    }

    function getSelectedSuggestion() {
      const pick = document.querySelector('input[name="suggPick"]:checked');
      const page = parseInt(document.getElementById('suggPage').value, 10);
      return { pick: pick ? parseInt(pick.value, 10) : 0, page };
    }

    async function applySuggestionToRow(mode) {
      const { pick, page } = getSelectedSuggestion();
      const labels = Array.from(document.querySelectorAll('#suggList label'));
      if (!labels[pick]) return;
      const cand = labels[pick].innerText.split(' (score')[0];
      const rowInput = document.getElementById('label_' + page);
      if (!rowInput) return;
      if (mode === 'apply') {
        setBusy(true, 'Applying...');
        rowInput.value = cand;
        setBusy(false);
        closeSuggest();
        return;
      }
      setBusy(true, 'Saving...');
      const saveRes = await fetch(urls.update(page), {
        method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ label: cand })
      });
      const saveData = await saveRes.json();
      if (!saveData.ok) {
        setBusy(false);
        alert('Save failed: ' + (saveData.error || ''));
        return;
      }
      rowInput.value = cand;
      if (mode === 'save') {
        setBusy(false);
        closeSuggest();
        location.reload();
        return;
      }
      const curateRes = await fetch(urls.curate(page), { method: 'POST' });
      const curateData = await curateRes.json();
      setBusy(false);
      if (!curateData.ok) {
        alert('Curate failed: ' + (curateData.error || ''));
        return;
      }
      closeSuggest();
      alert(mode === 'merge_curate' ? 'Merged alias, saved, and curated' : 'Saved and curated');
    }

    async function mergeAndCurate(urlAliasMerge) {
      const { pick, page } = getSelectedSuggestion();
      const labels = Array.from(document.querySelectorAll('#suggList label'));
      if (!labels[pick]) return;
      const cand = labels[pick].innerText.split(' (score')[0];
      const current = (document.getElementById('label_' + page)?.value || '').trim();
      if (!current) return alert('No current label');
      setBusy(true, 'Merging, saving and curating...');
      const m = await fetch(urlAliasMerge, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ alias: current, canonical: cand })
      });
      const md = await m.json();
      if (!md.ok) {
        setBusy(false);
        return alert('Merge failed: ' + (md.error || ''));
      }
      await applySuggestionToRow('merge_curate');
    }

    document.querySelectorAll('[data-open-suggest]').forEach((btn) => {
      const page = parseInt(btn.getAttribute('data-open-suggest'), 10);
      btn.addEventListener('click', () => openSuggest(page));
    });

    document.getElementById('suggClose')?.addEventListener('click', () => closeSuggest());
    document.getElementById('suggApply')?.addEventListener('click', () => applySuggestionToRow('apply'));
    document.getElementById('suggSave')?.addEventListener('click', () => applySuggestionToRow('save'));
    document.getElementById('suggCurate')?.addEventListener('click', () => applySuggestionToRow('curate'));
    document.getElementById('suggMergeCurate')?.addEventListener('click', () => mergeAndCurate(urls.aliasMerge));
  }

  document.addEventListener('DOMContentLoaded', () => {
    const urlsEl = document.getElementById('editUrls');
    if (!urlsEl) return;
    const urls = JSON.parse(urlsEl.textContent || '{}');

    const helpers = {
      update: (page) => urls.update.replace('__PAGE__', page),
      restore: (page) => urls.restore.replace('__PAGE__', page),
      curate: (page) => urls.curate.replace('__PAGE__', page),
      suggest: (page) => urls.suggest.replace('__PAGE__', page),
    };

    attachRowHandlers(helpers);

    const restoreAllBtn = document.querySelector('[data-restore-all]');
    restoreAllBtn?.addEventListener('click', () => restoreAll(urls.restore_all));

    const curateSelectedBtn = document.querySelector('[data-curate-selected]');
    curateSelectedBtn?.addEventListener('click', () => curateSelected(helpers));

    const curateAllBtn = document.querySelector('[data-curate-all]');
    curateAllBtn?.addEventListener('click', () => curateAll(urls.curate_all));

    const vectorSelect = document.getElementById('vectorSelect');
    if (vectorSelect) {
      loadVectorVersions(urls.vector_versions, 'vectorSelect', 'vectorActiveNote');
      vectorSelect.addEventListener('change', (event) => {
        const ver = event.target.value;
        if (ver) {
          setVectorVersion(urls.set_active_version, ver, 'vectorSelect', 'vectorActiveNote');
        }
      });
    }

    setupSuggest({
      suggest: helpers.suggest,
      update: helpers.update,
      curate: helpers.curate,
      aliasMerge: urls.alias_merge,
    });
  });
})();

