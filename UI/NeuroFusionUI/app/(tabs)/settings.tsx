import { useRouter } from 'expo-router';
import SettingsScreen from '../../src/screens/SettingsScreen';

export default function Settings() {
  const router = useRouter();

  const navigation = {
    goBack: () => router.back(),
    navigate: (screen: string) => {
      if (screen === 'AboutApp') {
        router.push('/aboutApp');
      } else if (screen === 'UserInformation') {
        router.push('./user-information');
      } else if (screen === 'Login') {
        router.replace('/login');
      }
    },
  };

  return <SettingsScreen/>;
}