import React, { useRef, useEffect } from 'react';
import {
  View, Text, StyleSheet, StatusBar,
  SafeAreaView, ScrollView, Animated, Linking, TouchableOpacity,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
 
const TEAM = [
  { name: 'Diadri Weerasekera', role: 'Developer & Researcher',   icon: 'account-circle' },
  { name: 'Dileeka Alwis',      role: 'Academic Supervisor',       icon: 'account-tie'    },
];
 
const MODEL_INFO = [
  { label: 'Architecture',  value: 'Dual-Branch CNN + MLP'  },
  { label: 'Training',      value: 'SupCon + Balanced'       },
  { label: 'Macro F1',      value: '0.2763'                  },
  { label: 'AUC-ROC',       value: '0.5016'                  },
  { label: 'MCC',           value: '+0.0064'                 },
  { label: 'McNemar p',     value: '0.0020'                  },
  { label: 'Window Size',   value: '256 samples'             },
  { label: 'Inference',     value: '< 50 ms on CPU'          },
];
 
const AboutAppScreen = () => {
  const router    = useRouter();
  const fadeAnim  = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(30)).current;
 
  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim,  { toValue: 1, duration: 500, useNativeDriver: true }),
      Animated.spring(slideAnim, { toValue: 0, friction: 8,   useNativeDriver: true }),
    ]).start();
  }, []);
 
  return (
    <SafeAreaView style={styles.container}>
      {/* <StatusBar barStyle="light-content" backgroundColor="#B844FF" /> */}
 
      {/* Header */}
      <Animated.View style={[styles.header, { opacity: fadeAnim }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <MaterialCommunityIcons name="arrow-left" size={22} color="#FFF" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>About App</Text>
        <View style={{ width: 36 }} />
      </Animated.View>
 
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnim }] }}>
 
          {/* App identity */}
          <View style={styles.heroCard}>
            <View style={styles.appIconWrap}>
              <MaterialCommunityIcons name="brain" size={52} color="#B844FF" />
            </View>
            <Text style={styles.appName}>EpiGuard</Text>
            <Text style={styles.appTagline}>AI-Powered EEG Seizure Detection</Text>
            <View style={styles.versionBadge}>
              <Text style={styles.versionText}>Version 1.0.0</Text>
            </View>
          </View>
 
          {/* About */}
          <View style={styles.card}>
            <View style={styles.cardTitleRow}>
              <MaterialCommunityIcons name="information" size={18} color="#B844FF" />
              <Text style={styles.cardTitle}>About EpiGuard</Text>
            </View>
            <Text style={styles.bodyText}>
              EpiGuard is a role-stratified patient monitoring application that uses
              a dual-branch deep learning model to analyse EEG signals and detect
              seizure activity. It is designed for patients, caretakers, and
              clinicians to collaboratively monitor epilepsy.
            </Text>
            <Text style={[styles.bodyText, { marginTop: 10 }]}>
              This application was developed as a Final Year Project at the
              Informatics Institute of Technology, Colombo, Sri Lanka (2025–2026),
              in affiliation with the University of Westminster.
            </Text>
          </View>
 
          {/* Model info */}
          <View style={styles.card}>
            <View style={styles.cardTitleRow}>
              <MaterialCommunityIcons name="chip" size={18} color="#B844FF" />
              <Text style={styles.cardTitle}>Model Details</Text>
            </View>
            {MODEL_INFO.map((item, i) => (
              <View key={item.label} style={[styles.infoRow, i === MODEL_INFO.length - 1 && { borderBottomWidth: 0 }]}>
                <Text style={styles.infoLabel}>{item.label}</Text>
                <Text style={styles.infoValue}>{item.value}</Text>
              </View>
            ))}
            <View style={styles.disclaimerBox}>
              <MaterialCommunityIcons name="alert-circle-outline" size={14} color="#F39C12" />
              <Text style={styles.disclaimerText}>
                For research purposes only. Not a certified medical device.
              </Text>
            </View>
          </View>
 
          {/* Team */}
          <View style={styles.card}>
            <View style={styles.cardTitleRow}>
              <MaterialCommunityIcons name="account-group" size={18} color="#B844FF" />
              <Text style={styles.cardTitle}>Development Team</Text>
            </View>
            {TEAM.map(member => (
              <View key={member.name} style={styles.teamRow}>
                <View style={styles.teamAvatar}>
                  <MaterialCommunityIcons name={member.icon} size={28} color="#B844FF" />
                </View>
                <View>
                  <Text style={styles.teamName}>{member.name}</Text>
                  <Text style={styles.teamRole}>{member.role}</Text>
                </View>
              </View>
            ))}
          </View>
 
          {/* Institution */}
          <View style={styles.card}>
            <View style={styles.cardTitleRow}>
              <MaterialCommunityIcons name="school" size={18} color="#B844FF" />
              <Text style={styles.cardTitle}>Institution</Text>
            </View>
            <Text style={styles.bodyText}>
              Informatics Institute of Technology (IIT){'\n'}
              In affiliation with the University of Westminster{'\n'}
              Colombo, Sri Lanka
            </Text>
          </View>
 
          <Text style={styles.footer}>© 2026 EpiGuard · All rights reserved</Text>
 
        </Animated.View>
      </ScrollView>
    </SafeAreaView>
  );
};
 
const styles = StyleSheet.create({
  container:   { flex: 1, backgroundColor: '#F4F5F9' },
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: '#B844FF', paddingHorizontal: 18, paddingVertical: 14,
  },
  backBtn: {
    width: 36, height: 36, borderRadius: 18,
    backgroundColor: 'rgba(255,255,255,0.2)',
    justifyContent: 'center', alignItems: 'center',
  },
  headerTitle: { fontSize: 18, fontWeight: '800', color: '#FFF' },
 
  scroll: { padding: 18, paddingBottom: 60 },
 
  heroCard: {
    backgroundColor: '#FFF', borderRadius: 20, padding: 28,
    alignItems: 'center', marginBottom: 14,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.07, shadowRadius: 8, elevation: 3,
  },
  appIconWrap: {
    width: 90, height: 90, borderRadius: 24,
    backgroundColor: '#F5EEFF',
    justifyContent: 'center', alignItems: 'center', marginBottom: 14,
  },
  appName:    { fontSize: 26, fontWeight: '800', color: '#1A1A2E', marginBottom: 4 },
  appTagline: { fontSize: 13, color: '#888', marginBottom: 12 },
  versionBadge: {
    backgroundColor: '#F5EEFF', borderRadius: 20,
    paddingHorizontal: 14, paddingVertical: 5,
  },
  versionText: { fontSize: 12, fontWeight: '700', color: '#B844FF' },
 
  card: {
    backgroundColor: '#FFF', borderRadius: 16, padding: 18, marginBottom: 14,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06, shadowRadius: 6, elevation: 2,
  },
  cardTitleRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 14 },
  cardTitle:    { fontSize: 15, fontWeight: '700', color: '#1A1A2E' },
  bodyText:     { fontSize: 13, color: '#555', lineHeight: 20 },
 
  infoRow: {
    flexDirection: 'row', justifyContent: 'space-between',
    paddingVertical: 9, borderBottomWidth: 1, borderBottomColor: '#F5F5F5',
  },
  infoLabel: { fontSize: 13, color: '#888' },
  infoValue: { fontSize: 13, fontWeight: '700', color: '#333' },
 
  disclaimerBox: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    backgroundColor: '#FFFBEB', borderRadius: 8, padding: 10, marginTop: 12,
  },
  disclaimerText: { flex: 1, fontSize: 11, color: '#92400E', lineHeight: 16 },
 
  teamRow: {
    flexDirection: 'row', alignItems: 'center', gap: 14,
    paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: '#F5F5F5',
  },
  teamAvatar: {
    width: 46, height: 46, borderRadius: 23,
    backgroundColor: '#F5EEFF',
    justifyContent: 'center', alignItems: 'center',
  },
  teamName: { fontSize: 14, fontWeight: '700', color: '#222' },
  teamRole: { fontSize: 12, color: '#888', marginTop: 2 },
 
  footer: { textAlign: 'center', fontSize: 11, color: '#CCC', marginTop: 10 },
});
 
export default AboutAppScreen;