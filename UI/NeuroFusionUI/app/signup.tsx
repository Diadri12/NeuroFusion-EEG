import { useRouter } from 'expo-router';
import SignUpScreen from '../src/screens/SignUpScreen';
import { Alert } from 'react-native';
import { createUserWithEmailAndPassword, updateProfile } from 'firebase/auth';
import { doc, setDoc } from 'firebase/firestore';
import { auth, db } from '../src/config/firebase';

export default function SignUp() {
  const router = useRouter();

  const handleSignUp = async (name: string, email: string, password: string, role: string) => {
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);

      await updateProfile(userCredential.user, { displayName: name });

      try {
        await setDoc(doc(db, 'users', userCredential.user.uid), {
          name,
          email,
          role,
          createdAt: new Date().toISOString(),
        });
      } catch (firestoreError) {
        console.log('Firestore save failed (non-blocking):', firestoreError);
      }

      // Save role to Firestore
      await setDoc(doc(db, 'users', userCredential.user.uid), {
        name,
        email,
        role,
        createdAt: new Date().toISOString(),
      });

      switch (role) {
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
      Alert.alert('Sign up failed', error.message);
    }
  };

  const handleNavigateToLogin = () => {
    router.back();
  };

  return (
    <SignUpScreen
      onSignUp={handleSignUp}
      onNavigateToLogin={handleNavigateToLogin}
    />
  );
}