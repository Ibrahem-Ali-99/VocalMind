import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { loginWithEmail, loginWithGoogle, getUserMe, logoutUser, User } from "../services/api";

interface AuthContextType {
  token: string | null;
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  googleLogin: (idToken: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchUser = async () => {
      try {
        const userData = await getUserMe();
        setUser(userData);
        setToken("cookie-based");
      } catch (e) {
        setUser(null);
        setToken(null);
      } finally {
        setIsLoading(false);
      }
    };
    fetchUser();
  }, []);

  const login = async (email: string, pass: string) => {
    await loginWithEmail(email, pass);
    const userData = await getUserMe();
    setUser(userData);
    setToken("cookie-based");
  };

  const googleLogin = async (idToken: string) => {
    await loginWithGoogle(idToken);
    const userData = await getUserMe();
    setUser(userData);
    setToken("cookie-based");
  };

  const logout = async () => {
    try {
      await logoutUser();
    } catch (e) {
      console.error("Logout error:", e);
    }
    setToken(null);
    setUser(null);
    window.location.href = "/login";
  };

  return (
    <AuthContext.Provider
      value={{
        token,
        user,
        isAuthenticated: !!token,
        isLoading,
        login,
        googleLogin,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
