import { View, Text, StyleSheet } from 'react-native';
import DoctorReportsScreen from '../../src/screens/DoctorReportsScreen'


export default function ReportsScreen() {
  return <DoctorReportsScreen/>
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#F8F9FA' },
  text:      { fontSize: 18, fontWeight: '600', color: '#333' },
});