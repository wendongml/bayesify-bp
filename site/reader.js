// Real links remain usable without JavaScript. Preserve the reading topic when
// a visitor chooses another implementation; never load all three notebooks.
const tabs = [...document.querySelectorAll('.framework')];
function selectTopic(id) {
  for (const tab of tabs) {
    const url = new URL(tab.href); url.hash = id; tab.href = url.href;
  }
  for (const link of document.querySelectorAll('aside nav a')) {
    link.classList.toggle('active', link.hash === '#' + id);
  }
}
if (location.hash) selectTopic(location.hash.slice(1));
const chapters = [...document.querySelectorAll('h2.chapter')];
let pending = false;
addEventListener('scroll', () => {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    let selected = chapters[0];
    for (const chapter of chapters) {
      if (chapter.getBoundingClientRect().top < innerHeight * .4) selected = chapter;
    }
    if (selected) selectTopic(selected.id);
    pending = false;
  });
}, {passive: true});
if (innerWidth < 900) document.querySelector('aside details').open = false;
