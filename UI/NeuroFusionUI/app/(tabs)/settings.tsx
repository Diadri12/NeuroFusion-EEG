import { useRouter } from 'expo-router';
import { signOut } from 'firebase/auth';
import { auth } from '../../src/config/firebase';
import SettingsScreen from '../../src/screens/SettingsScreen';
 
export default function Settings() {
  const router = useRouter();
 
  const handleSignOut = async () => {
    await signOut(auth);
  };
 
  const handleNavigate = (screen: string) => {
    if (screen === 'userInformation') router.push('/userInformation');
    if (screen === 'aboutApp')        router.push('/aboutApp');
  };
 
  return (
    <SettingsScreen
      onSignOut={handleSignOut}
      onNavigate={handleNavigate}
    />
  );
}