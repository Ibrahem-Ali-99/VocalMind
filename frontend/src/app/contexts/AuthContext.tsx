import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { loginWithEmail, loginWithGoogle, getUserMe, User } from "../services/api";

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
    const fetchUser = async (storedToken: string) => {
      try {
        const payload = JSON.parse(atob(storedToken.split(".")[1]));
        const isExpired = payload.exp ? Date.now() >= payload.exp * 1000 : false;
        
        if (isExpired) {
          localStorage.removeItem("vocalmind_token");
          setToken(null);
          setUser(null);
        } else {
          setToken(storedToken);
          // Fetch full user profile since we have a valid token
          try {
            const userData = await getUserMe();
            setUser(userData);
          } catch (e) {
            // Fallback if API fails
            setUser({ email: payload.sub || "user@vocalmind.ai" } as any);
          }
        }
      } catch (e) {
        console.error("Auth initialization error:", e);
      } finally {
        setIsLoading(false);
      }
    };

    const storedToken = localStorage.getItem("vocalmind_token");
    if (storedToken && storedToken !== "null" && storedToken !== "undefined") {
      fetchUser(storedToken);
    } else {
      setIsLoading(false);
    }
  }, []);

  const login = async (email: string, pass: string) => {
    const { access_token } = await loginWithEmail(email, pass);
    localStorage.setItem("vocalmind_token", access_token);
    setToken(access_token);
    
    // Fetch full user profile after login
    const userData = await getUserMe();
    setUser(userData);
  };

  const googleLogin = async (idToken: string) => {
    const { access_token } = await loginWithGoogle(idToken);
    localStorage.setItem("vocalmind_token", access_token);
    setToken(access_token);
    
    const userData = await getUserMe();
    setUser(userData);
  };

  const logout = () => {
    localStorage.removeItem("vocalmind_token");
    setToken(null);
    setUser(null);
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
