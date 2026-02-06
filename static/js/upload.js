(() => {
  const busy = document.getElementById('busy');
  const busyMsg = document.getElementById('busyMsg');
  function showBusy(msg) {
    if (!busy || !busyMsg) return;
    busyMsg.textContent = msg || 'Working...';
    busy.style.display = 'flex';
  }
  function hideBusy() {
    if (!busy) return;
    busy.style.display = 'none';
  }
  window.addEventListener('pageshow', hideBusy);
  const form = document.querySelector('[data-upload-form]');
  if (form) {
    form.addEventListener('submit', () => {
      showBusy('Generating mapping...');
    });
  }
})();

