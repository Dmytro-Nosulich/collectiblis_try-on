/**
 * The mode-chooser screen: three tiles (Live Try-On, Use a Photo, Choose a
 * Model), shown first, per docs/step2-design-decisions.md. Use a
 * Photo/Choose a Model are visibly disabled stubs until Phases 9-10 build
 * them.
 */

export interface ModeChooserHandlers {
  onLiveTryOn(): void;
}

const PRIVACY_LINE = 'Nothing is uploaded — this stays on your device.';

interface TileConfig {
  title: string;
  description: string;
  disabled: boolean;
  onSelect?: () => void;
}

function createTile(config: TileConfig): HTMLButtonElement {
  const tile = document.createElement('button');
  tile.type = 'button';
  tile.className = config.disabled ? 'mode-tile mode-tile--disabled' : 'mode-tile';
  tile.disabled = config.disabled;

  const title = document.createElement('p');
  title.className = 'mode-tile-title';
  title.append(config.title);
  if (config.disabled) {
    const badge = document.createElement('span');
    badge.className = 'mode-tile-badge';
    badge.textContent = 'Coming soon';
    title.append(badge);
  }

  const description = document.createElement('p');
  description.className = 'mode-tile-desc';
  description.textContent = config.description;

  tile.append(title, description);
  if (config.onSelect) {
    tile.addEventListener('click', config.onSelect);
  }
  return tile;
}

/** Clears `container` and renders the mode chooser into it. */
export function renderModeChooser(container: HTMLElement, handlers: ModeChooserHandlers): void {
  container.replaceChildren();

  const heading = document.createElement('h2');
  heading.className = 'mode-chooser-title';
  heading.textContent = 'How would you like to try this on?';

  const tiles = document.createElement('div');
  tiles.className = 'mode-tiles';
  tiles.append(
    createTile({
      title: 'Live Try-On',
      description: 'Use your camera to see it on you in real time.',
      disabled: false,
      onSelect: handlers.onLiveTryOn,
    }),
    createTile({
      title: 'Use a Photo',
      description: 'Upload a photo of yourself instead.',
      disabled: true,
    }),
    createTile({
      title: 'Choose a Model',
      description: "Don't want to use your own photo? Pick one of ours.",
      disabled: true,
    }),
  );

  const privacyLine = document.createElement('p');
  privacyLine.className = 'privacy-line';
  privacyLine.textContent = PRIVACY_LINE;

  container.append(heading, tiles, privacyLine);
}
