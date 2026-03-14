import { useRouter } from 'expo-router';
import DashboardScreen from '../../src/screens/DashboardScreen';

export default function Dashboard() {
  const router = useRouter();

  const handleNavigateToAnalyze = () => {
    router.push('/(tabs)/analyzeEEG');
  };

  return <DashboardScreen onNavigateToAnalyze={handleNavigateToAnalyze} />;
}