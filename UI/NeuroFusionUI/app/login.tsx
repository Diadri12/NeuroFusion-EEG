import { useRouter } from 'expo-router';
import LoginScreen from '../src/screens/LoginScreen';
import { Alert } from 'react-native';
import { signInWithEmailAndPassword } from 'firebase/auth';
import { doc, getDoc } from 'firebase/firestore';
import { auth, db } from '../src/config/firebase';

export default function Login() {
  const router = useRouter();

  const handleLogin = async (email: string, password: string, role: string) => {
    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);

      // Optionally verify role matches what's stored in Firestore
      const userDoc = await getDoc(doc(db, 'users', userCredential.user.uid));
      const storedRole = userDoc.exists() ? userDoc.data().role : role;

      switch (storedRole) {
        case 'caretaker':
          router.replace('/(tabs)/caretakerDashboard');
          break;
        case 'doctor':
          router.replace('/(tabs)/doctorDashboard');
          break;
        default:
          router.replace('/(tabs)/dashboard');
      }
    } catch (error: any) {
      Alert.alert('Login failed', error.message);
    }
  };

  const handleNavigateToSignUp = () => {
    router.push('/signup');
  };

  return (
    <LoginScreen
      onLogin={handleLogin}
      onNavigateToSignUp={handleNavigateToSignUp}
    />
  );
}