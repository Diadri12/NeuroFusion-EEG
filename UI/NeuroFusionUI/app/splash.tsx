import { useEffect } from 'react';
import { useRouter } from 'expo-router';
import SplashScreen from '../src/screens/SplashScreen';

export default function Splash() {
  const router = useRouter();

  useEffect(() => {
    const timer = setTimeout(() => {
      router.replace('/login');
    }, 5000);

    return () => clearTimeout(timer);
  }, []);

  return <SplashScreen />;
}