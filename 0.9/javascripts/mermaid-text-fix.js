// Force black text in mermaid diagrams for dark mode (light backgrounds need dark text)
document$.subscribe(function() {
  const isDarkMode = document.querySelector('[data-md-color-scheme="slate"]') !== null;

  if (isDarkMode) {
    // Wait for mermaid to render
    setTimeout(function() {
      const mermaidTexts = document.querySelectorAll('.mermaid svg text, .mermaid svg tspan');
      mermaidTexts.forEach(function(el) {
        el.setAttribute('fill', '#000000');
        el.style.fill = '#000000';
      });
    }, 100);
  }
});

// Also run on theme toggle
const observer = new MutationObserver(function(mutations) {
  mutations.forEach(function(mutation) {
    if (mutation.attributeName === 'data-md-color-scheme') {
      const isDarkMode = document.querySelector('[data-md-color-scheme="slate"]') !== null;

      if (isDarkMode) {
        const mermaidTexts = document.querySelectorAll('.mermaid svg text, .mermaid svg tspan');
        mermaidTexts.forEach(function(el) {
          el.setAttribute('fill', '#000000');
          el.style.fill = '#000000';
        });
      }
    }
  });
});

observer.observe(document.body, {
  attributes: true,
  attributeFilter: ['data-md-color-scheme']
});
