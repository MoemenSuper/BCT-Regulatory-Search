import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import Root from './Root.tsx';
import { clearGlobalThemeAttribute } from './hooks/useTheme';

clearGlobalThemeAttribute();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
