import { useCallback, useEffect, useState } from 'react';
import { getMe, logout, type AuthUser } from './api/auth';
import { AccountStatusPage } from './components/AccountStatusPage';
import { AdminDashboard } from './components/AdminDashboard';
import { LoginPage } from './components/LoginPage';
import { RegisterPage } from './components/RegisterPage';
import { type UiLocale } from './uiLocale';
import App from './App';
import './loginPageStyle.css';
import './adminStyles.css';
import './adminAppleStyles.css';

type Gate =
  | { kind: 'loading' }
  | { kind: 'login' }
  | { kind: 'register' }
  | { kind: 'session'; user: AuthUser };

export default function Root() {
  const [gate, setGate] = useState<Gate>({ kind: 'loading' });
  const [locale, setLocale] = useState<UiLocale>(() => {
    const saved = window.localStorage.getItem('bct-ui-locale');
    return saved === 'ar' || saved === 'en' ? saved : 'fr';
  });

  function handleLocaleChange(nextLocale: UiLocale) {
    window.localStorage.setItem('bct-ui-locale', nextLocale);
    setLocale(nextLocale);
  }

  const loadSession = useCallback(async () => {
    try {
      const result = await getMe();
      setGate({ kind: 'session', user: result.user });
    } catch {
      setGate({ kind: 'login' });
    }
  }, []);

  useEffect(() => {
    void loadSession();
  }, [loadSession]);

  async function handleLogout() {
    try {
      await logout();
    } catch {
      // Session may already be gone.
    }
    setGate({ kind: 'login' });
  }

  if (gate.kind === 'loading') {
    return <div className="auth-loading">Chargement…</div>;
  }

  if (gate.kind === 'login') {
    return (
      <LoginPage
        onAuthenticated={(user) => setGate({ kind: 'session', user })}
        onGoRegister={() => setGate({ kind: 'register' })}
        locale={locale}
        onLocaleChange={handleLocaleChange}
      />
    );
  }

  if (gate.kind === 'register') {
    return (
      <RegisterPage
        onRegistered={(user) => setGate({ kind: 'session', user })}
        onGoLogin={() => setGate({ kind: 'login' })}
        locale={locale}
        onLocaleChange={handleLocaleChange}
      />
    );
  }

  const { user } = gate;
  if (user.status !== 'approved') {
    return <AccountStatusPage user={user} onLogout={() => void handleLogout()} locale={locale} onLocaleChange={handleLocaleChange} />;
  }
  if (user.role === 'admin') {
    return <AdminDashboard user={user} onLogout={() => void handleLogout()} locale={locale} onLocaleChange={handleLocaleChange} />;
  }
  return <App onLogout={() => void handleLogout()} />;
}
