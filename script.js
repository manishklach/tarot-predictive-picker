const navLinks = Array.from(document.querySelectorAll(".nav a"));
const sections = Array.from(document.querySelectorAll("main section[id]"));
const revealTargets = Array.from(document.querySelectorAll(".card, .section-heading, .timeline-step"));
const yearNode = document.getElementById("year");

if (yearNode) {
  yearNode.textContent = new Date().getFullYear();
}

revealTargets.forEach((element) => {
  element.setAttribute("data-reveal", "");
});

if ("IntersectionObserver" in window) {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.16,
      rootMargin: "0px 0px -40px 0px"
    }
  );

  revealTargets.forEach((element) => revealObserver.observe(element));

  const navObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

      if (!visible) {
        return;
      }

      const activeId = visible.target.id;

      navLinks.forEach((link) => {
        const isActive = link.getAttribute("href") === `#${activeId}`;
        link.classList.toggle("active", isActive);
      });
    },
    {
      threshold: 0.35,
      rootMargin: "-25% 0px -55% 0px"
    }
  );

  sections.forEach((section) => navObserver.observe(section));
} else {
  revealTargets.forEach((element) => element.classList.add("is-visible"));
}
